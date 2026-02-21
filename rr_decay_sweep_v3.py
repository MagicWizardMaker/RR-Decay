import argparse, json, math, os
import numpy as np
import matplotlib.pyplot as plt

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
    return np.argsort(v, kind="mergesort")[:k]  # deterministic tie-breaking

def k_smallest_action_from_theta(theta_hat: np.ndarray, k: int):
    idx = topk_min_indices_stable(theta_hat, k)
    a = np.zeros_like(theta_hat)
    a[idx] = 1.0
    return a, idx

def ci95_from_std(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)

def wilson_ci95(k: int, n: int):
    # Wilson score interval for binomial proportion
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    rad = (z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))

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

# -----------------------
# Uniform top-k covariance (closed form)
# -----------------------
def uniform_topk_cov_params(d: int, k: int):
    p = k / d
    rho = (k * (k - 1)) / (d * (d - 1))
    alpha = p - rho
    return p, rho, alpha

def theta_hat_cov_ridge(bvec: np.ndarray, n: int, d: int, k: int, lam: float) -> np.ndarray:
    """
    theta_hat_n = (lam I + n G)^(-1) b_n
    where G = (p-rho) I + rho 11^T (uniform top-k).
    Closed form inverse via Sherman-Morrison, O(d).
    """
    if n <= 0:
        return np.zeros(d, dtype=float)
    p, rho, alpha = uniform_topk_cov_params(d, k)
    A = lam + n * alpha          # scalar
    B = n * rho                  # scalar
    denom = A + B * d            # scalar
    s = float(bvec.sum())
    return (bvec / A) - (B / (A * denom)) * s * np.ones(d, dtype=float)

# -----------------------
# Environment
# -----------------------
def generate_theta_fixed_best(
    T, d, star_support, low_range, high_range, drift_std,
    burst_prob=0.0, burst_len=0, burst_gap=None
):
    """
    star coords drift in [low_min, low_max], non-star drift in [high_min, high_max],
    except in bursts where non-star coords are pinned to target = low_max + burst_gap
    (boundary near-ties).
    """
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

        # near-tie burst: pin outside coords near boundary low_max + burst_gap
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
# RR-decay with prefix-separation proxy tracking
# -----------------------
def run_rr_decay(theta, star_support, c_eps, lam, noise, eps_grid, tie_tol=1e-12):
    """
    Returns:
      per-run summary + gamma_hist,
      and (for Tmax pooling) the prefix proxy curve phi_n = min_{i out, j in} D_n(i,j)/n over exploration index n.
    """
    T, d = theta.shape
    k = int(star_support.sum())

    a_star = np.zeros(d, dtype=float)
    a_star[star_support] = 1.0

    # exploration stats
    bvec = np.zeros(d, dtype=float)
    n_exp = 0
    theta_hat = np.zeros(d, dtype=float)
    greedy_action, _ = k_smallest_action_from_theta(theta_hat, k)

    # boundary pairs
    S_idx = np.where(star_support)[0]
    Sc_idx = np.where(~star_support)[0]
    D = np.zeros((len(Sc_idx), len(S_idx)), dtype=float)  # D_n(i,j) matrix over boundary pairs

    # record phi_n curve along exploration index
    phi_curve = []

    regret = np.zeros(T, dtype=float)
    greedy_ok_hist = np.zeros(T, dtype=int)
    gamma_hist = np.zeros(T, dtype=float)

    wrong_exploit = 0
    exploit_cnt = 0
    wrong_all = 0
    wrong_ge = {float(eps): 0 for eps in eps_grid}
    tie_cnt = 0

    for t in range(T):
        th = theta[t]
        g = swap_margin(th, star_support)
        gamma_hist[t] = g
        if g <= tie_tol:
            tie_cnt += 1

        # evaluate current greedy (policy state before choosing action this round)
        greedy_ok = int(np.all(greedy_action == a_star))
        greedy_ok_hist[t] = greedy_ok
        if not greedy_ok:
            wrong_all += 1
            for eps in eps_grid:
                if g >= eps:
                    wrong_ge[float(eps)] += 1

        eps_t = min(1.0, c_eps / math.sqrt(t + 1))
        explore = (np.random.rand() < eps_t)

        if explore:
            a = sample_uniform_k_action(d, k)
            eta = float(np.random.uniform(-noise, noise)) if noise > 0 else 0.0
            y = float(a @ th + eta)

            bvec += a * y
            n_exp += 1

            # update estimator
            theta_hat = theta_hat_cov_ridge(bvec=bvec, n=n_exp, d=d, k=k, lam=lam)
            greedy_action, _ = k_smallest_action_from_theta(theta_hat, k)

            # update D_n(i,j) and record phi_n
            D += (th[Sc_idx][:, None] - th[S_idx][None, :])
            phi_curve.append(float((D / n_exp).min()))
        else:
            exploit_cnt += 1
            if not greedy_ok:
                wrong_exploit += 1
            a = greedy_action

        regret[t] = float((a - a_star) @ th)

    # horizon-lock-in time tau: first round after which greedy equals a* for all remaining rounds up to T
    ok_from = np.ones(T + 1, dtype=bool)
    ok_from[T] = True
    for t in range(T - 1, -1, -1):
        ok_from[t] = ok_from[t + 1] and (greedy_ok_hist[t] == 1)
    tau = T + 1
    for t in range(T):
        if ok_from[t]:
            tau = t + 1
            break

    quantiles = {str(q): float(np.quantile(gamma_hist, q)) for q in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]}

    res = {
        "R": float(regret.sum()),
        "R_over_sqrtT": float(regret.sum() / math.sqrt(T)),
        "n_explore": int(n_exp),
        "tau": int(tau),                 # tau=T+1 means "not locked by horizon"
        "locked_by_T": bool(tau <= T),
        "wrong_frac_policy_all": float(wrong_all / T),
        "wrong_frac_policy_exploit": float(wrong_exploit / max(exploit_cnt, 1)),
        "wrong_ge_frac_policy_all": {str(eps): float(wrong_ge[float(eps)] / T) for eps in eps_grid},
        "tie_frac": float(tie_cnt / T),
        "min_gamma": float(gamma_hist.min()),
        "quantiles": quantiles,
        "exploit_cnt": int(exploit_cnt),
    }
    return res, gamma_hist, np.array(phi_curve, dtype=float)

# -----------------------
# Sweep + plots (adds prefix-separation proxy plot at Tmax)
# -----------------------
def sweep(args):
    d, k = args.d, args.k
    Ts = [int(x) for x in args.Ts.split(",")]
    eps_grid = tuple(float(x) for x in args.eps_grid.split(","))

    os.makedirs(args.outdir, exist_ok=True)

    p, rho, alpha = uniform_topk_cov_params(d, k)

    meta = {
        "d": d, "k": k, "seeds": args.seeds, "Ts": Ts,
        "mode": args.mode,
        "c_eps": args.c_eps, "lam": args.lam,
        "noise_model": "uniform",
        "noise": args.noise,
        "burst_params": {"burst_prob": args.burst_prob, "burst_len": args.burst_len, "burst_gap": args.burst_gap},
        "theta_params": {
            "low_min": args.low_min, "low_max": args.low_max,
            "high_min": args.high_min, "high_max": args.high_max,
            "drift_std": args.drift_std,
        },
        "covariance": {"p": p, "rho": rho, "alpha": alpha},
        "eps_grid": list(eps_grid),
        "burst_type": "boundary_near_tie_bursts",
    }

    curve = {
        "T": [],
        "R_mean": [], "R_std": [], "R_ci95": [],
        "R_over_sqrtT_mean": [], "R_over_sqrtT_std": [], "R_over_sqrtT_ci95": [],
        "explore_mean": [], "explore_std": [], "explore_ci95": [],
        "tau_mean": [], "tau_std": [], "tau_ci95": [],
        "lock_rate": [], "locked_count": [], "lock_rate_ci95": [],
        "wrong_exploit_mean": [], "wrong_exploit_std": [], "wrong_exploit_ci95": [],
    }

    Tmax = max(Ts)
    pooled_gamma_Tmax = []
    pooled_phi_curves = []  # store phi_n curves for Tmax

    for T in Ts:
        R_list = []
        Rrs_list = []
        explore_list = []
        tau_list = []
        lock_list = []
        wrong_ex_list = []

        for sd in range(args.seeds):
            set_seed(sd)

            # sample S* directly (avoid enumerating all actions)
            star_idx = np.random.choice(d, size=k, replace=False)
            star_support = np.zeros(d, dtype=bool)
            star_support[star_idx] = True

            theta = generate_theta_fixed_best(
                T=T, d=d, star_support=star_support,
                low_range=(args.low_min, args.low_max),
                high_range=(args.high_min, args.high_max),
                drift_std=args.drift_std,
                burst_prob=(args.burst_prob if args.mode == "bursts" else 0.0),
                burst_len=(args.burst_len if args.mode == "bursts" else 0),
                burst_gap=(args.burst_gap if args.mode == "bursts" else None),
            )

            out, gamma_hist, phi_curve = run_rr_decay(
                theta=theta,
                star_support=star_support,
                c_eps=args.c_eps,
                lam=args.lam,
                noise=args.noise,
                eps_grid=eps_grid,
            )

            R_list.append(out["R"])
            Rrs_list.append(out["R_over_sqrtT"])
            explore_list.append(out["n_explore"])
            tau_list.append(out["tau"])
            lock_list.append(1 if out["locked_by_T"] else 0)
            wrong_ex_list.append(out["wrong_frac_policy_exploit"])

            if T == Tmax:
                pooled_gamma_Tmax.append(gamma_hist)
                pooled_phi_curves.append(phi_curve)

        # aggregate for this T
        R = np.array(R_list, dtype=float)
        Rrs = np.array(Rrs_list, dtype=float)
        E = np.array(explore_list, dtype=float)
        Tau = np.array(tau_list, dtype=float)
        Lock = np.array(lock_list, dtype=int)
        Wex = np.array(wrong_ex_list, dtype=float)

        curve["T"].append(T)

        curve["R_mean"].append(float(R.mean()))
        curve["R_std"].append(float(R.std(ddof=1) if len(R) > 1 else 0.0))
        curve["R_ci95"].append(float(ci95_from_std(curve["R_std"][-1], len(R))))

        curve["R_over_sqrtT_mean"].append(float(Rrs.mean()))
        curve["R_over_sqrtT_std"].append(float(Rrs.std(ddof=1) if len(Rrs) > 1 else 0.0))
        curve["R_over_sqrtT_ci95"].append(float(ci95_from_std(curve["R_over_sqrtT_std"][-1], len(Rrs))))

        curve["explore_mean"].append(float(E.mean()))
        curve["explore_std"].append(float(E.std(ddof=1) if len(E) > 1 else 0.0))
        curve["explore_ci95"].append(float(ci95_from_std(curve["explore_std"][-1], len(E))))

        curve["tau_mean"].append(float(Tau.mean()))
        curve["tau_std"].append(float(Tau.std(ddof=1) if len(Tau) > 1 else 0.0))
        curve["tau_ci95"].append(float(ci95_from_std(curve["tau_std"][-1], len(Tau))))

        locked_count = int(Lock.sum())
        lock_rate = float(locked_count / len(Lock))
        (lo, hi) = wilson_ci95(locked_count, len(Lock))
        curve["locked_count"].append(locked_count)
        curve["lock_rate"].append(lock_rate)
        curve["lock_rate_ci95"].append([float(lo), float(hi)])

        curve["wrong_exploit_mean"].append(float(Wex.mean()))
        curve["wrong_exploit_std"].append(float(Wex.std(ddof=1) if len(Wex) > 1 else 0.0))
        curve["wrong_exploit_ci95"].append(float(ci95_from_std(curve["wrong_exploit_std"][-1], len(Wex))))

    # pooled margin diagnostics at Tmax
    if pooled_gamma_Tmax:
        pooled = np.concatenate(pooled_gamma_Tmax, axis=0)
        md = {
            "Tmax": Tmax,
            "pooled_quantiles": {str(q): float(np.quantile(pooled, q)) for q in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]},
            "below_frac": {str(e): float(np.mean(pooled <= e)) for e in eps_grid},
            "max_run": {str(e): int(max_run_leq(pooled, e)) for e in eps_grid},
        }
    else:
        md = None

    # prefix-separation proxy curve at Tmax: align phi curves by exploration index
    prefix_proxy = None
    if pooled_phi_curves:
        max_n = max(len(c) for c in pooled_phi_curves)
        M = np.full((len(pooled_phi_curves), max_n), np.nan, dtype=float)
        for r, c in enumerate(pooled_phi_curves):
            M[r, :len(c)] = c

        mean_phi = np.nanmean(M, axis=0)
        std_phi = np.nanstd(M, axis=0, ddof=1)
        n_eff = np.sum(~np.isnan(M), axis=0)
        ci_phi = np.array([ci95_from_std(std_phi[i], int(n_eff[i])) for i in range(max_n)], dtype=float)

        prefix_proxy = {
            "Tmax": Tmax,
            "n_index": list(range(1, max_n + 1)),
            "phi_mean": mean_phi.tolist(),
            "phi_ci95": ci_phi.tolist(),
            "n_eff": n_eff.astype(int).tolist(),
            "phi_min_mean_over_prefix": float(np.nanmin(mean_phi)),
            "phi_final_mean": float(mean_phi[~np.isnan(mean_phi)][-1]),
        }

        # plot (this is a diagnostic figure; you likely won't include it in the paper)
        plt.figure()
        x = np.arange(1, max_n + 1)
        plt.plot(x, mean_phi, label=r"$\phi_n=\min_{i\notin S^\star,j\in S^\star} D_n(i,j)/n$")
        plt.fill_between(x, mean_phi - ci_phi, mean_phi + ci_phi, alpha=0.2, label="95% CI")
        plt.xlabel("Exploration index n")
        plt.ylabel(r"$\phi_n$")
        plt.title(f"Prefix-separation proxy along exploration times (T={Tmax}, mode={args.mode})")
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"prefix_sep_proxy_{args.mode}.png"), dpi=180)
        plt.close()

    out = {
        "meta": meta,
        "curve": curve,
        "margin_diagnostics_at_Tmax": md,
        "prefix_sep_proxy_at_Tmax": prefix_proxy,
    }

    json_path = os.path.join(args.outdir, f"rr_decay_{args.mode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # standard plots (means only; you will use at most 2 in paper)
    Ts_arr = np.array(curve["T"], dtype=float)
    Rrs = np.array(curve["R_over_sqrtT_mean"], dtype=float)
    Rrs_ci = np.array(curve["R_over_sqrtT_ci95"], dtype=float)

    plt.figure()
    plt.plot(Ts_arr, Rrs, marker="o", label="mean")
    plt.fill_between(Ts_arr, np.maximum(Rrs - Rrs_ci, 0.0), Rrs + Rrs_ci, alpha=0.2, label="95% CI")
    plt.xscale("log")
    plt.xlabel("T")
    plt.ylabel(r"$R_T/\sqrt{T}$")
    plt.title(f"Scaling check (mode={args.mode}, c={args.c_eps})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"regret_over_sqrtT_{args.mode}.png"), dpi=180)
    plt.close()

    print(f"[OK] wrote {json_path} and plots to {args.outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--Ts", type=str, default="200,500,1000,2000,5000,10000,20000,50000")
    ap.add_argument("--seeds", type=int, default=200)

    ap.add_argument("--mode", type=str, default="constant", choices=["constant", "bursts"])
    ap.add_argument("--c_eps", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--noise", type=float, default=0.0)

    # theta ranges
    ap.add_argument("--low_min", type=float, default=0.02)
    ap.add_argument("--low_max", type=float, default=0.035)
    ap.add_argument("--high_min", type=float, default=0.195)
    ap.add_argument("--high_max", type=float, default=0.23)
    ap.add_argument("--drift_std", type=float, default=0.0025)

    # bursts params
    ap.add_argument("--burst_prob", type=float, default=5e-5)
    ap.add_argument("--burst_len", type=int, default=500)
    ap.add_argument("--burst_gap", type=float, default=0.0)

    ap.add_argument("--eps_grid", type=str, default="0.0,0.001,0.01,0.05,0.1")
    ap.add_argument("--outdir", type=str, default="out_rr_decay_v3")

    args = ap.parse_args()
    sweep(args)

if __name__ == "__main__":
    main()