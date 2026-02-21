# MagicWizardMaker — RR-decay (Nonstationary Top-k Linear Bandit)

This repository contains the simulation code and experiment outputs used to evaluate **RR-decay** in a nonstationary **top-k linear bandit** with a **stable optimal set** and **near-tie burst** episodes.

The experiments stress-test:

1. Whether regret is consistent with $\widetilde{O}(\sqrt{T})$ scaling under decaying exploration,
2. Whether the algorithm **locks in** to the correct top-k set under prolonged **near-tie bursts**, and
3. Whether performance is robust to **bounded observation noise**.

---

## Problem setting (simulation)

- **Dimension:** `d = 12`
- **Top-k size:** `k = 4`
- **Default horizon grid:**

$$
T \in \{200, 500, 1000, 2000, 5000, 10^4, 2\cdot 10^4, 5\cdot 10^4\}
$$

- Loss vectors $\theta_t \in \mathbb{R}^d$ are **time-varying**
- A fixed optimal set $S^\star$ (size $k$) is sampled per seed
- Actions are indicator vectors of size-$k$ subsets, and loss is $\langle a_t,\theta_t\rangle$

### Near-tie bursts

In `mode=bursts`, with small probability a burst of length `500` starts. During a burst:

- Coordinates **outside** $S^\star$ are pinned to the boundary value `low_max`
- Coordinates **inside** $S^\star$ continue drifting in `[low_min, low_max]`

This creates long contiguous segments where the instantaneous boundary gap

$$
\gamma_t = \min_{i\notin S^\star}\theta_{t,i} - \max_{j\in S^\star}\theta_{t,j}
$$

becomes **near zero** (and may be exactly zero up to numerical precision), while preserving the stable-optimum benchmark.

### Noise model

When `--noise 0.02`, feedback uses bounded uniform noise:

$$
y_t = \langle a_t, \theta_t \rangle + \eta_t,
\qquad
\eta_t \sim \mathrm{Unif}[-0.02, 0.02]
$$

---

## Algorithm (RR-decay)

At each round $t$:

- **Explore** with probability

$$
\varepsilon_t = \min\{1, c/\sqrt{t}\}
$$

by sampling a **uniform top-k** action.

- Otherwise **exploit** by choosing the $k$ smallest coordinates of the current estimate.

The estimator is a **covariance-aware ridge** estimator, updated only on exploration rounds:

$$
\hat\theta_n = (\lambda I + nG)^{-1} b_n,
\qquad
b_n = \sum_{s=1}^n a_{t_s} y_{t_s}
$$

where $G = \mathbb{E}[aa^\top]$ is the known covariance for uniform top-k actions.

---

## Code

### `rr_decay_sweep_v2.py`

Sweep runner that:

- Writes JSON summaries per run
- Produces standard plots:
  - regret log–log
  - $R/\sqrt{T}$
  - wrong fraction
  - exploration budget
- Includes confidence intervals and slope-fit utilities used in paper drafts

### `rr_decay_sweep_v3.py`

Extends v2 with an additional diagnostic aligned with the paper’s identifiability condition, computed **along exploration times**:

$$
\phi_n := \min_{i\notin S^\star,\,j\in S^\star}
\frac{1}{n}\sum_{s=1}^n (\theta_{t_s,i}-\theta_{t_s,j})
$$

This adds an extra diagnostic plot:

- `prefix_sep_proxy_<mode>.png`

---

## How to run

All commands below use `--seeds 200`.

### Recommended (paper-grade) runs

#### Near-tie bursts, noiseless

```bash
python rr_decay_sweep_v3.py --mode bursts --c_eps 3 --noise 0.0  --seeds 200 --outdir out_bursts_c3
python rr_decay_sweep_v3.py --mode bursts --c_eps 5 --noise 0.0  --seeds 200 --outdir out_bursts_c5
```

#### Near-tie bursts, bounded noise

```bash
python rr_decay_sweep_v3.py --mode bursts --c_eps 3 --noise 0.02 --seeds 200 --outdir out_bursts_noise_c3
python rr_decay_sweep_v3.py --mode bursts --c_eps 5 --noise 0.02 --seeds 200 --outdir out_bursts_noise_c5
```

#### Optional baseline (constant regime)

```bash
python rr_decay_sweep_v3.py --mode constant --c_eps 5 --noise 0.0 --seeds 200 --outdir out_const_rr_c5_200
```

### Custom horizon grid

You can change the horizon grid via `--Ts`, for example:

```bash
python rr_decay_sweep_v3.py --Ts 2000,5000,10000,20000,50000 --mode bursts --c_eps 5 --seeds 200 --outdir out_custom
```

---

## Output format

Each output folder contains:

- `rr_decay_<mode>.json` — metadata + curve summaries
- `regret_over_sqrtT_<mode>.png` — normalized regret scaling plot (mean ± 95% CI)
- `prefix_sep_proxy_<mode>.png` — exploration-time prefix separation proxy (**v3 only**)

### JSON contents

The JSON summary includes:

- Means, standard deviations, and 95% confidence intervals
- Exploration counts $N_T$
- Horizon lock-in time $\tau$ (with $\tau = T + 1$ if not locked)
- Lock rate over seeds
- Wrong-on-exploit rate

---

## Repository structure

This repo includes both:

- **Scripts:** `rr_decay_sweep_v2.py`, `rr_decay_sweep_v3.py`
- **Saved outputs:** `out_*` folders used for paper figures/tables and reproducibility

If you only want to reproduce the paper tables/figures, start from the `out_*` folders.  
If you want to regenerate everything, use the commands above.

---

## Reproducibility notes

- Randomness is controlled by `--seeds` via `numpy.random.seed(seed)` per seed
- Tie-breaking for greedy top-k is deterministic (stable sort), which avoids ambiguity when estimated coordinates tie
- The burst mechanism is designed to create prolonged boundary near-ties (not necessarily exact ties every round)

---

## License

Add your preferred license (MIT / BSD / Apache-2.0 / etc.) if you plan to make this public.

---

## Contact

Add author names/emails here (or remove this section for anonymous review).
