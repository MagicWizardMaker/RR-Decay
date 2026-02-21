import argparse, json, math, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    np.random.seed(seed)

def sample_uniform_k_action(d: int, k: int) -> np.ndarray:
    idx = np.random.choice(d, size=k, replace=False)
    a = np.zeros(d, dtype=float)
    a[idx] = 1.0
    return a

def topk_min_indices_stable(v: np.ndarray, k: int) -> np.ndarray:
    # Stable tie-breaking: smaller index wins
    return np.argsort(v, kind="mergesort")[:k]

def k_smallest_action_from_theta(theta_hat: np.ndarray, k: int):
    idx = topk_min_indices_stable(theta_hat, k)
    a = np.zeros_like(theta_hat)
    a[idx] = 1.0
    return a, idx

def ci95(xs: np.ndarray) -> float:
    if len(xs) <= 1:
        return 0.0
    return 1.96 * float(xs.std(ddof=1)) / math.sqrt(len(xs))

# -----------------------
# Uniform top-k covariance G = alpha I + rho 11^T
# -----------------------
def uniform_topk_covariance(d: int, k: int):
    p = k / d
    rho = (k * (k - 1)) / (d * (d - 1))
    alpha = p - rho
    # eigenvalues: alpha (mult d-1), alpha + rho d (along 1)
    lam_min = alpha
    cond = (alpha + rho * d) / max(alpha, 1e-30)
    return dict(p=p, rho=rho, alpha=alpha, lam_min=lam_min, cond=cond)

def cov_ridge_theta_hat(bvec: np.ndarray, n: int, d: int, k: int, lam: float) -> np.ndarray:
    """
    Closed-form for (lam I + nG)^(-1) bvec where G = alpha I + rho 11^T.
    (lam I + nG) = A I + B 11^T, with A = lam + n alpha, B = n rho.
    Inverse: (1/A)I - (B / (A (A + Bd))) 11^T.
    """
    if n <= 0:
        return np.zeros(d, dtype=float)

    cov = uniform_topk_covariance(d, k)
    alpha = cov["alpha"]
    rho = cov["rho"]

    A = lam + n * alpha
    B = n * rho
    denom = A + B * d  # = lam + n(alpha + rho d)

    s = float(bvec.sum())
    return (bvec / A) - (B / (A * denom)) * s * np.ones(d, dtype=float)

# -----------------------
# Environment
# -----------------------
def generate_theta_fixed_best(
    T, d, star_support, low_range, high_range, drift_std,
    burst_prob=0.0, burst_len=0, burst_gap=None
):
    low_min, low_max = low_range
    high_min, high_max = high_range
    assert low_max < high_min

    theta = np.zeros((T, d), dtype=float)
    th0 = np.zeros(d, dtype=float)
    th0[star_support]  = np.random.uniform(low_min,  low_max,  size=int(star_support.sum()))
    th0[~star_support] = np.random.uniform(high_min, high_max, size=int((~star_support).sum()))
    theta[0] = th0

    t = 1
    while t < T:
        th = theta[t - 1].copy() + np.random.normal(0.0, drift_std, size=d)

        # Burst: pin outside coords to boundary level low_max + burst_gap
        if burst_prob > 0 and np.random.rand() < burst_prob and burst_len > 0:
            gap = float(burst_gap) if burst_gap is not None else float(high_min - low_max)
            target = low_max + gap
            for tt in range(t, min(T, t + burst_len)):
                thb = theta[tt - 1].copy() + np.random.normal(0.0, drift_std, size=d)
                for i in range(d):
                    if star_support[i]:
                        a, b = low_min, low_max
                        val = thb[i]
                        if val < a: val = a + (a - val)
                        if val > b: val = b - (val - b)
                        thb[i] = min(max(val, a), b)
                    else:
                        thb[i] = target
                theta[tt] = thb
            t = min(T, t + burst_len)
            continue

        # reflect + clip
        for i in range(d):
            a, b = (low_min, low_max) if star_support[i] else (high_min, high_max)
            val = th[i]
            if val < a: val = a + (a - val)
            if val > b: val = b - (val - b)
            th[i] = min(max(val, a), b)

        theta[t] = th
        t += 1

    return theta

def swap_margin(theta_t: np.ndarray, star_support: np.ndarray) -> float:
    S = np.where(star_support)[0]
    Sc = np.where(~star_support)[0]
    return float(theta_t[Sc].min() - theta_t[S].max())

# -----------------------
# RR-decay (covariance-aware ridge)
# -----------------------
def run_rr_decay_cov(
    theta: np.ndarray,
    star_support: np.ndarray,
    c_eps: float,
    lam: float,
    noise_model: str,
    noise: float,
    eps_grid=(0.0, 0.001, 0.01, 0.05, 0.1),
):
    T, d = theta.shape
    k = int(star_support.sum())

    a_star = np.zeros(d, dtype=float)
    a_star[star_support] = 1.0

    # sufficient statistic on exploration samples only
    bvec = np.zeros(d, dtype=float)
    n = 0
    theta_hat = np.zeros(d, dtype=float)
    greedy_action, _ = k_smallest_action_from_theta(theta_hat, k)

    regret = 0.0
    explore_cnt = 0
    wrong_all = 0
    wrong_exploit = 0
    exploit_cnt = 0

    wrong_ge = {float(eps): 0 for eps in eps_grid}

    # Margin tracking (all rounds) for reporting
    gamma_hist = np.zeros(T, dtype=float)

    # Prefix-separation proxy along exploration times:
    # track min over boundary pairs of D_n(i,j)/n, and its minimum over n
    S_idx = np.where(star_support)[0]
    Sc_idx = np.where(~star_support)[0]
    pair_sum = np.zeros((len(Sc_idx), len(S_idx)), dtype=float)
    min_prefix_avg = +1e18

    greedy_ok_hist = np.zeros(T, dtype=int)

    for t in range(T):
        th = theta[t]
        g = swap_margin(th, star_support)
        gamma_hist[t] = g

        # policy wrongness at start of round (before sampling action)
        is_greedy_ok = int(np.all(greedy_action == a_star))
        greedy_ok_hist[t] = is_greedy_ok
        if not is_greedy_ok:
            wrong_all += 1
            for eps in eps_grid:
                if g >= eps:
                    wrong_ge[float(eps)] += 1

        eps_t = min(1.0, c_eps / math.sqrt(t + 1))
        explore = (np.random.rand() < eps_t)

        if explore:
            a = sample_uniform_k_action(d, k)
            explore_cnt += 1

            if noise_model == "uniform":
                eta = float(np.random.uniform(-noise, noise)) if noise > 0 else 0.0
            else:
                raise ValueError("noise_model must be 'uniform' in this script.")

            y = float(a @ th + eta)

            bvec += a * y
            n += 1

            # update theta_hat and greedy only on exploration
            theta_hat = cov_ridge_theta_hat(bvec, n, d, k, lam)
            greedy_action, _ = k_smallest_action_from_theta(theta_hat, k)

            # update exploration-prefix diagnostic
            # D_n(i,j) accumulates theta_{t,i} - theta_{t,j} along exploration times
            # We maintain all boundary pairs; d=12 so this is cheap.
            pair_sum += (th[Sc_idx][:, None] - th[S_idx][None, :])
            cur_min = float((pair_sum / n).min())
            min_prefix_avg = min(min_prefix_avg, cur_min)

        else:
            exploit_cnt += 1
            if not is_greedy_ok:
                wrong_exploit += 1
            a = greedy_action
            y = float(a @ th)  # observed but unused (kept for conceptual consistency)

        regret += float((a - a_star) @ th)

    # lock-in time tau: first t such that greedy is a_star for all rounds >= t
    ok_from = np.ones(T + 1, dtype=bool)
    ok_from[T] = True
    for t in range(T - 1, -1, -1):
        ok_from[t] = ok_from[t + 1] and (greedy_ok_hist[t] == 1)
    tau = T + 1
    for t in range(T):
        if ok_from[t]:
            tau = t + 1
            break

    res = dict(
        R=float(regret),
        explore=int(explore_cnt),
        tau=int(tau),
        lock=int(tau <= T),
        wrong_policy_all=float(wrong_all / T),
        wrong_policy_exploit=float(wrong_exploit / max(exploit_cnt, 1)),
        min_gamma=float(gamma_hist.min()),
        gamma_quantiles={str(q): float(np.quantile(gamma_hist, q)) for q in [0.0,0.01,0.05,0.5,0.95,0.99,1.0]},
        wrong_ge_frac={str(eps): float(wrong_ge[float(eps)] / T) for eps in eps_grid},
        explore_min_prefix_avg=float(min_prefix_avg if n > 0 else 0.0),
        explore_final_prefix_avg=float((pair_sum / n).min() if n > 0 else 0.0),
    )
    return res

# -----------------------
# Sweep + plots
# -----------------------
def fit_slope_loglog(Ts, Rm, points_used):
    xs = np.log(np.array(points_used, dtype=float))
    ys = np.log(np.array([Rm[Ts.index(t)] for t in points_used], dtype=float))
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope), float(intercept)

def sweep(args):
    os.makedirs(args.outdir, exist_ok=True)

    Ts = [int(x) for x in args.Ts.split(",")]
    eps_grid = tuple(float(x) for x in args.eps_grid.split(","))

    meta = dict(
        d=args.d, k=args.k, seeds=args.seeds, Ts=Ts, mode=args.mode,
        c_eps=args.c_eps, lam=args.lam,
        noise_model=args.noise_model, noise=args.noise,
        theta_params=dict(
            low_min=args.low_min, low_max=args.low_max,
            high_min=args.high_min, high_max=args.high_max,
            drift_std=args.drift_std,
            burst_prob=args.burst_prob, burst_len=args.burst_len, burst_gap=args.burst_gap,
        ),
        eps_grid=list(eps_grid),
        covariance=uniform_topk_covariance(args.d, args.k),
    )

    curve = {k: [] for k in [
        "T",
        "R_mean","R_std","R_ci95",
        "R_over_sqrtT_mean","R_over_sqrtT_std","R_over_sqrtT_ci95",
        "wrong_policy_all_mean","wrong_policy_all_std","wrong_policy_all_ci95",
        "wrong_policy_exploit_mean","wrong_policy_exploit_std","wrong_policy_exploit_ci95",
        "explore_mean","explore_std","explore_ci95",
        "tau_mean","tau_std","tau_ci95",
        "lock_rate",
        "min_gamma_mean","min_gamma_std","min_gamma_ci95",
        "explore_min_prefix_avg_mean","explore_min_prefix_avg_std","explore_min_prefix_avg_ci95",
        "explore_final_prefix_avg_mean","explore_final_prefix_avg_std","explore_final_prefix_avg_ci95",
    ]}

    # For pooled margin diagnostics at Tmax
    Tmax = max(Ts)
    pooled_gamma = []

    # Choose star set by sampling k indices (no need to enumerate all actions)
    for T in Ts:
        Rs=[]; wrongA=[]; wrongE=[]; explores=[]; taus=[]; locks=[]; mins=[]; pmin=[]; pfinal=[]
        for sd in range(args.seeds):
            set_seed(sd)

            star_idx = np.random.choice(args.d, size=args.k, replace=False)
            star_support = np.zeros(args.d, dtype=bool)
            star_support[star_idx] = True

            theta = generate_theta_fixed_best(
                T=T, d=args.d, star_support=star_support,
                low_range=(args.low_min, args.low_max),
                high_range=(args.high_min, args.high_max),
                drift_std=args.drift_std,
                burst_prob=(args.burst_prob if args.mode=="bursts" else 0.0),
                burst_len=(args.burst_len if args.mode=="bursts" else 0),
                burst_gap=(args.burst_gap if args.mode=="bursts" else None),
            )

            out = run_rr_decay_cov(
                theta=theta,
                star_support=star_support,
                c_eps=args.c_eps,
                lam=args.lam,
                noise_model=args.noise_model,
                noise=args.noise,
                eps_grid=eps_grid,
            )

            Rs.append(out["R"])
            wrongA.append(out["wrong_policy_all"])
            wrongE.append(out["wrong_policy_exploit"])
            explores.append(out["explore"])
            taus.append(out["tau"])
            locks.append(out["lock"])
            mins.append(out["min_gamma"])
            pmin.append(out["explore_min_prefix_avg"])
            pfinal.append(out["explore_final_prefix_avg"])

            if T == Tmax:
                # Pool gamma_t across seeds for diagnostics
                g = np.array([swap_margin(theta[t], star_support) for t in range(T)], dtype=float)
                pooled_gamma.append(g)

        Rs=np.array(Rs); wrongA=np.array(wrongA); wrongE=np.array(wrongE)
        explores=np.array(explores); taus=np.array(taus); locks=np.array(locks)
        mins=np.array(mins); pmin=np.array(pmin); pfinal=np.array(pfinal)

        curve["T"].append(T)

        curve["R_mean"].append(float(Rs.mean()))
        curve["R_std"].append(float(Rs.std(ddof=1)))
        curve["R_ci95"].append(ci95(Rs))

        Rs_sqrt = Rs / math.sqrt(T)
        curve["R_over_sqrtT_mean"].append(float(Rs_sqrt.mean()))
        curve["R_over_sqrtT_std"].append(float(Rs_sqrt.std(ddof=1)))
        curve["R_over_sqrtT_ci95"].append(ci95(Rs_sqrt))

        curve["wrong_policy_all_mean"].append(float(wrongA.mean()))
        curve["wrong_policy_all_std"].append(float(wrongA.std(ddof=1)))
        curve["wrong_policy_all_ci95"].append(ci95(wrongA))

        curve["wrong_policy_exploit_mean"].append(float(wrongE.mean()))
        curve["wrong_policy_exploit_std"].append(float(wrongE.std(ddof=1)))
        curve["wrong_policy_exploit_ci95"].append(ci95(wrongE))

        curve["explore_mean"].append(float(explores.mean()))
        curve["explore_std"].append(float(explores.std(ddof=1)))
        curve["explore_ci95"].append(ci95(explores))

        curve["tau_mean"].append(float(taus.mean()))
        curve["tau_std"].append(float(taus.std(ddof=1)))
        curve["tau_ci95"].append(ci95(taus))

        curve["lock_rate"].append(float(locks.mean()))

        curve["min_gamma_mean"].append(float(mins.mean()))
        curve["min_gamma_std"].append(float(mins.std(ddof=1)))
        curve["min_gamma_ci95"].append(ci95(mins))

        curve["explore_min_prefix_avg_mean"].append(float(pmin.mean()))
        curve["explore_min_prefix_avg_std"].append(float(pmin.std(ddof=1)))
        curve["explore_min_prefix_avg_ci95"].append(ci95(pmin))

        curve["explore_final_prefix_avg_mean"].append(float(pfinal.mean()))
        curve["explore_final_prefix_avg_std"].append(float(pfinal.std(ddof=1)))
        curve["explore_final_prefix_avg_ci95"].append(ci95(pfinal))

    # slope fit on last 4 points (if available)
    points_used = Ts[-4:] if len(Ts) >= 4 else Ts
    slope, intercept = fit_slope_loglog(Ts, curve["R_mean"], points_used)
    slope_fit = dict(points_used=points_used, slope_loglog=slope)

    # pooled margin diagnostics at Tmax
    if pooled_gamma:
        pooled = np.concatenate(pooled_gamma, axis=0)
        md = dict(
            Tmax=Tmax,
            pooled_quantiles={str(q): float(np.quantile(pooled, q)) for q in [0.0,0.01,0.05,0.5,0.95,0.99,1.0]},
            below_frac={str(e): float(np.mean(pooled <= e)) for e in eps_grid},
            max_run={str(e): int(max_run_leq(pooled, e)) for e in eps_grid},
        )
    else:
        md = None

    out = dict(meta=meta, curve=curve, slope_fit=slope_fit, margin_diagnostics_at_Tmax=md)
    out["fixed_best_all"] = True  # by construction of the generator

    json_path = os.path.join(args.outdir, f"rr_decay_{args.mode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # plots (means + error bars)
    Ts_arr = np.array(curve["T"], dtype=float)
    Rm = np.array(curve["R_mean"], dtype=float)
    Rci = np.array(curve["R_ci95"], dtype=float)
    Rrs = np.array(curve["R_over_sqrtT_mean"], dtype=float)
    Rrs_ci = np.array(curve["R_over_sqrtT_ci95"], dtype=float)
    Wm = np.array(curve["wrong_policy_all_mean"], dtype=float)
    Wci = np.array(curve["wrong_policy_all_ci95"], dtype=float)
    Em = np.array(curve["explore_mean"], dtype=float)
    Eci = np.array(curve["explore_ci95"], dtype=float)

    plt.figure()
    plt.errorbar(Ts_arr, Rm, yerr=Rci, fmt="o-")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("T"); plt.ylabel("Mean regret R(T)")
    plt.title(f"RR-decay regret (mode={args.mode}, c={args.c_eps})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"regret_loglog_{args.mode}.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(Ts_arr, Rrs, yerr=Rrs_ci, fmt="o-")
    plt.xscale("log")
    plt.xlabel("T"); plt.ylabel("R(T) / sqrt(T)")
    plt.title(f"Scaling check (mode={args.mode}, c={args.c_eps})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"regret_over_sqrtT_{args.mode}.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(Ts_arr, Wm, yerr=Wci, fmt="o-")
    plt.xscale("log")
    plt.xlabel("T"); plt.ylabel("Wrong fraction (greedy != a*)")
    plt.title(f"Identification quality (mode={args.mode}, c={args.c_eps})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"wrong_frac_{args.mode}.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(Ts_arr, Em, yerr=Eci, fmt="o-")
    plt.xscale("log")
    plt.xlabel("T"); plt.ylabel("Mean #exploration samples")
    plt.title(f"Exploration budget (c={args.c_eps})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"explore_budget_{args.mode}.png"), dpi=180)
    plt.close()

    print(f"[OK] wrote {json_path} and plots to {args.outdir}")

def max_run_leq(x: np.ndarray, eps: float) -> int:
    run = 0
    best = 0
    for v in x:
        if v <= eps:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--Ts", type=str, default="200,500,1000,2000,5000,10000,20000,50000")
    ap.add_argument("--seeds", type=int, default=200)

    ap.add_argument("--mode", type=str, default="constant", choices=["constant","bursts"])
    ap.add_argument("--c_eps", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)

    ap.add_argument("--noise_model", type=str, default="uniform", choices=["uniform"])
    ap.add_argument("--noise", type=float, default=0.0)

    # theta ranges
    ap.add_argument("--low_min", type=float, default=0.02)
    ap.add_argument("--low_max", type=float, default=0.035)
    ap.add_argument("--high_min", type=float, default=0.195)
    ap.add_argument("--high_max", type=float, default=0.23)
    ap.add_argument("--drift_std", type=float, default=0.0025)

    # bursts params (used only if mode=bursts)
    ap.add_argument("--burst_prob", type=float, default=5e-5)
    ap.add_argument("--burst_len", type=int, default=500)
    ap.add_argument("--burst_gap", type=float, default=0.0)

    ap.add_argument("--eps_grid", type=str, default="0.0,0.001,0.01,0.05,0.1")
    ap.add_argument("--outdir", type=str, default="out_rr_decay")

    args = ap.parse_args()
    sweep(args)

if __name__ == "__main__":
    main()