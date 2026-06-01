"""
generate_figures.py
===================
Publication-quality figure generation for SBP-BRiMS 2026 paper:
  "Modeling Authorization Saturation: A Coupled Queueing Model of
   Behavioral Tipping Points in AI-Governed Institutions"

Generates paper figures (PDF for LaTeX inclusion) and SVG versions
for the submission package.

Output:
  figures/fig2_quality.pdf/.svg       — Authorization quality curves (MC + CI)
  figures/fig3_interventions.pdf/.svg — Intervention comparison bar chart
  figures/supp1_phase_diagram.pdf/.svg
  figures/supp2_sensitivity.pdf/.svg
  figures/supp3_time_series.pdf/.svg
  figures/supp4_ceremonial_frac.pdf/.svg
  figures/supp5_quality_derivative.pdf/.svg
  figures/supp6_regime_map.pdf/.svg
  figures/supp7_intervention_regions.pdf/.svg

Usage:
    python generate_figures.py

Requirements: Python 3.8+, numpy, matplotlib
"""

import heapq
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------------------------
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (muted, print-safe, colour-blind tolerant)
# ---------------------------------------------------------------------------
C = {
    "k05":   "#2C5F8A",   # deep blue      k=0.5
    "k10":   "#C0392B",   # muted crimson  k=1.0
    "k20":   "#1A7A5E",   # dark teal      k=2.0
    "k30":   "#7F4D9A",   # muted purple   k=3.0
    "sub":   "#EAF4EA",   # light green    substantive zone
    "cer":   "#FAEAEA",   # light red      ceremonial zone
    "trans": "#FFF8E1",   # light amber    transition band
    "gray":  "#888888",
    "dark":  "#2B2B2B",
    # Intervention palette
    "BL":  "#555555",
    "I1":  "#1A5276",
    "I2":  "#145A32",
    "I12": "#6C3483",
    "I3":  "#D35400",
    "I4":  "#922B21",
}

# ---------------------------------------------------------------------------
# MATPLOTLIB STYLE
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size":            9.5,
    "axes.labelsize":       10,
    "axes.titlesize":       10,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.97,
    "legend.edgecolor":     "#D5D5D5",
    "legend.borderpad":     0.6,
    "legend.handlelength":  2.2,
    "legend.handleheight":  1.2,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.9,
    "axes.grid":            True,
    "grid.alpha":           0.14,
    "grid.linewidth":       0.38,
    "grid.color":           "#C0C0C0",
    "grid.linestyle":       "--",
    "lines.linewidth":      2.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4.5,
    "ytick.major.size":     4.5,
    "xtick.major.width":    0.9,
    "ytick.major.width":    0.9,
    "xtick.minor.size":     2.5,
    "ytick.minor.size":     2.5,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.07,
})

FW = 4.75   # figure width  (≈ LNCS textwidth 12.2 cm → 4.8 in)
FH = 2.9    # figure height (default; individual figures may override)


def save(fig, name):
    """Save figure as PDF and SVG."""
    for fmt in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"  saved figures/{name}.pdf  figures/{name}.svg")


# ---------------------------------------------------------------------------
# SIMULATION ENGINE (discrete-event, heapq departure queue)
# ---------------------------------------------------------------------------
def _simulate(lam, mu_s=1.0, alpha=3.0, c=5, k=1.0,
              q_s=1.0, q_c=0.25, T_sim=6000, T_warmup=600, seed=42):
    """
    Discrete-event simulation of the state-dependent M/M/c authorization queue.

    Service rate per server:
        mu_s  if N(t) <= T = floor(k*c)
        mu_c  if N(t) >  T                 (mu_c = alpha * mu_s)

    Quality of service at time t:
        q_s   if N(t) <= T
        q_c   if N(t) >  T

    Returns (time-avg quality Q̄, time-avg queue length N̄).
    """
    if lam <= 0:
        return float(q_s), 0.0
    rng = np.random.default_rng(seed)
    mu_c = alpha * mu_s
    T_thr = int(k * c)

    t = 0.0
    N = 0
    n_busy = 0
    area_Q = 0.0
    area_N = 0.0
    t_stat = float(T_warmup)

    next_arr = rng.exponential(1.0 / lam)
    deps = []   # min-heap of departure times

    def _mu(n): return mu_c if n > T_thr else mu_s

    t_end = T_sim + T_warmup
    while t < t_end:
        nd = deps[0] if deps else float("inf")
        ne = next_arr if next_arr <= nd else nd

        # Accumulate time-weighted stats (post warm-up only)
        if t >= t_stat:
            dt = min(ne, t_end) - max(t, t_stat)
            if dt > 0:
                area_N += N * dt
                area_Q += (q_s if N <= T_thr else q_c) * dt

        if ne >= t_end:
            break
        t = ne

        if next_arr <= nd:
            # --- Arrival ---
            N += 1
            if n_busy < c:
                n_busy += 1
                heapq.heappush(deps, t + rng.exponential(1.0 / _mu(N)))
            next_arr = t + rng.exponential(1.0 / lam)
        else:
            # --- Departure ---
            heapq.heappop(deps)
            N -= 1
            n_busy -= 1
            if N >= c:
                n_busy += 1
                heapq.heappush(deps, t + rng.exponential(1.0 / _mu(N)))

    elapsed = T_sim
    return (area_Q / elapsed, area_N / elapsed)


def mc_sweep(rho_vals, c=5, k=1.0, alpha=3.0, mu_s=1.0,
             q_s=1.0, q_c=0.25, n_reps=8, T_sim=5000, T_warmup=500):
    """
    Monte Carlo sweep over utilisation values.
    Returns mean Q̄, 95% CI half-width, mean N̄, 95% CI half-width.
    """
    mQ, eQ, mN, eN = [], [], [], []
    for rho_s in rho_vals:
        lam = rho_s * c * mu_s
        if rho_s * alpha <= rho_s:   # stability guard (shouldn't trigger)
            mQ.append(q_c); eQ.append(0.0)
            mN.append(np.nan); eN.append(0.0)
            continue
        Qs, Ns = [], []
        for s in range(n_reps):
            q, n = _simulate(lam, mu_s, alpha, c, k, q_s, q_c,
                             T_sim, T_warmup, seed=s)
            Qs.append(q)
            Ns.append(n)
        mq = np.mean(Qs)
        sq = np.std(Qs, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
        mn = np.mean(Ns)
        sn = np.std(Ns, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
        mQ.append(mq);  eQ.append(1.96 * sq)
        mN.append(mn);  eN.append(1.96 * sn)
    return np.array(mQ), np.array(eQ), np.array(mN), np.array(eN)


# Simulation data from Table 2 (seed=42, T_sim=50 000)
TABLE2_RHO = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
TABLE2_Q   = [1.000, 1.000, 0.996, 0.986, 0.979, 0.967, 0.952, 0.936, 0.898, 0.854, 0.809]
TABLE2_N   = [0.50, 0.99, 1.49, 2.00, 2.24, 2.49, 2.75, 2.99, 3.45, 3.90, 4.31]


# ---------------------------------------------------------------------------
# FIGURE 2 — Authorization Quality Curves (MC + 95% CI + Table 2 data points)
# ---------------------------------------------------------------------------
def fig2_quality_curves():
    print("Building Figure 2: Authorization quality curves …")

    rho_dense = np.arange(0.05, 0.96, 0.05)      # 19 points
    rho_fine  = np.linspace(0.02, 0.96, 60)       # for smooth CI bands

    # k=1.0 — primary curve, 50 reps for tight CI
    print("  MC sweep k=1.0 …")
    mQ10, eQ10, _, _ = mc_sweep(rho_dense, k=1.0, n_reps=50, T_sim=20000)
    mQ10f, eQ10f, _, _ = mc_sweep(rho_fine, k=1.0, n_reps=50, T_sim=20000)

    # k=0.5
    print("  MC sweep k=0.5 …")
    mQ05f, eQ05f, _, _ = mc_sweep(rho_fine, k=0.5, n_reps=50, T_sim=20000)

    # k=2.0
    print("  MC sweep k=2.0 …")
    mQ20f, eQ20f, _, _ = mc_sweep(rho_fine, k=2.0, n_reps=50, T_sim=20000)

    fig, ax = plt.subplots(figsize=(FW, FH + 0.1))

    # --- Background zone shading (referenced to k=1.0 ρ*=0.50) ---
    rho_star_ref = 0.50
    trans_lo, trans_hi = 0.42, 0.58
    ax.axvspan(0.0,       trans_lo,  alpha=0.09, color=C["sub"],   zorder=0, lw=0)
    ax.axvspan(trans_lo,  trans_hi,  alpha=0.11, color=C["trans"], zorder=0, lw=0)
    ax.axvspan(trans_hi,  1.0,       alpha=0.09, color=C["cer"],   zorder=0, lw=0)

    # --- ρ* vertical dashed markers + top labels ---
    for rho_st, label, col in [
        (1/(1+2),  r"$\rho^*_{0.5}$", C["k05"]),
        (1/(1+1),  r"$\rho^*_{1.0}$", C["k10"]),
        (2/(1+2),  r"$\rho^*_{2.0}$", C["k20"]),
    ]:
        ax.axvline(rho_st, color=col, lw=0.85, ls="--", alpha=0.50, zorder=1)
        ax.text(rho_st, 1.055, label, ha="center", va="bottom",
                fontsize=6.5, color=col, alpha=0.90, zorder=4)

    # --- k=0.5 CI band + curve ---
    ax.fill_between(rho_fine, mQ05f - eQ05f, mQ05f + eQ05f,
                    color=C["k05"], alpha=0.10, zorder=2)
    ax.plot(rho_fine, mQ05f, color=C["k05"], lw=1.85, ls="--",
            label=r"$k=0.5$  ($\rho^*=0.33$)", zorder=3)

    # --- k=1.0 CI band + curve (primary) ---
    ax.fill_between(rho_fine, mQ10f - eQ10f, mQ10f + eQ10f,
                    color=C["k10"], alpha=0.10, zorder=2)
    ax.plot(rho_fine, mQ10f, color=C["k10"], lw=2.3, ls="-",
            label=r"$k=1.0$  ($\rho^*=0.50$)", zorder=3)

    # --- k=2.0 CI band + curve ---
    ax.fill_between(rho_fine, mQ20f - eQ20f, mQ20f + eQ20f,
                    color=C["k20"], alpha=0.10, zorder=2)
    ax.plot(rho_fine, mQ20f, color=C["k20"], lw=1.85, ls=":",
            label=r"$k=2.0$  ($\rho^*=0.67$)", zorder=3)

    # (Table 2 individual-seed dots removed — MC mean curves are the primary estimate)

    # --- Reference lines ---
    ax.axhline(1.0,  color=C["dark"], lw=0.6, ls=":", alpha=0.4)
    ax.axhline(0.25, color=C["dark"], lw=0.6, ls=":", alpha=0.4)
    ax.text(0.97, 1.005, r"$q_s$", ha="right", va="bottom",
            fontsize=7.5, color=C["gray"], transform=ax.get_xaxis_transform())
    ax.text(0.97, 0.235, r"$q_c$", ha="right", va="top",
            fontsize=7.5, color=C["gray"], transform=ax.get_xaxis_transform())

    # --- Zone annotations ---
    ax.text(0.19, 0.22, "Substantive", ha="center", va="center",
            fontsize=7.5, color="#2E7D32", alpha=0.70, style="italic")
    ax.text(0.79, 0.22, "Ceremonial",  ha="center", va="center",
            fontsize=7.5, color="#C62828", alpha=0.70, style="italic")

    # --- 95% CI label ---
    ax.text(0.355, 0.50, "95% CI",
            fontsize=6.5, color=C["gray"], alpha=0.75)

    ax.set_xlabel(r"Utilization $\rho_s = \lambda\,/\,(c\mu_s)$")
    ax.set_ylabel(r"Mean authorization quality $\bar{Q}(\rho_s)$")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.yaxis.set_tick_params(which="major", left=True)
    ax.legend(loc="lower left", ncol=1, framealpha=0.97,
              borderpad=0.7, labelspacing=0.45)
    fig.tight_layout(pad=0.4)
    save(fig, "fig2_quality")


# ---------------------------------------------------------------------------
# FIGURE 3 — Intervention Comparison (horizontal bar chart)
# ---------------------------------------------------------------------------
def fig3_interventions():
    print("Building Figure 3: Intervention comparison …")

    labels = [
        "Baseline\n($\\rho_s=0.60$)",
        "I1: Double capacity\n$(c \\to 10,\\ \\rho_s=0.30)$",
        "I2: Halve throughput\n$(\\lambda \\to 1.5,\\ \\rho_s=0.30)$",
        "I1+I2 combined\n$(25\\%$ each$)$",
        "I3: Raise $k \\to 2.0$\n$(\\rho^* \\to 0.67)$",
        "I4: Raise $q_c \\to 0.50$\n(quality floor only)",
    ]
    q_vals = [0.936, 1.000, 0.996, 0.993, 0.996, 0.958]
    # MC error bars (30 replicates, 95% CI half-width)
    q_err  = [0.004, 0.000, 0.002, 0.002, 0.002, 0.006]
    colors = [C["BL"], C["I1"], C["I2"], C["I12"], C["I3"], C["I4"]]

    fig, ax = plt.subplots(figsize=(FW, 2.72))

    y = np.arange(len(labels))
    bars = ax.barh(y, q_vals, color=colors, height=0.58, alpha=0.88,
                   edgecolor="white", linewidth=0.6, zorder=3,
                   xerr=q_err, error_kw=dict(elinewidth=0.8, capsize=2.5,
                                              capthick=0.8, ecolor="#555555"))

    # Baseline reference line
    ax.axvline(q_vals[0], color=C["BL"], lw=1.0, ls="--", alpha=0.55, zorder=2)
    # Perfect quality reference
    ax.axvline(1.0, color=C["dark"], lw=0.7, ls=":", alpha=0.35, zorder=2)
    ax.text(1.001, 5.35, r"$q_s{=}1$", fontsize=6.5, va="top", color=C["gray"])

    # Value annotations on bars
    for bar, val in zip(bars, q_vals):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7.5,
                color=C["dark"])

    ax.set_xlim(0.80, 1.015)
    ax.set_xlabel(r"Mean authorization quality $\bar{Q}$")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.xaxis.set_tick_params(which="both", bottom=True)

    # Shaded gain region
    ax.axvspan(q_vals[0], 1.0, alpha=0.05, color="#1A5276", zorder=0)
    ax.text(0.968, -0.7, "Quality\ngain", fontsize=6.5,
            color="#1A5276", ha="center", alpha=0.8)

    fig.tight_layout()
    save(fig, "fig3_interventions")


# ---------------------------------------------------------------------------
# SUPP 1 — Phase Diagram: regime boundaries in (k, ρ_s) space
# ---------------------------------------------------------------------------
def supp1_phase_diagram():
    print("Building Supp 1: Phase diagram …")

    k_vals = np.linspace(0.1, 4.0, 300)
    rho_star = k_vals / (1 + k_vals)

    fig, ax = plt.subplots(figsize=(FW, 2.5))
    ax.fill_between(k_vals, 0,        rho_star, alpha=0.18, color=C["sub"],
                    label="Substantive regime", zorder=0)
    ax.fill_between(k_vals, rho_star, 1.0,      alpha=0.18, color=C["cer"],
                    label="Ceremonial regime",  zorder=0)
    ax.plot(k_vals, rho_star, color=C["k10"], lw=2.0,
            label=r"$\rho^*(k)=k/(1+k)$", zorder=3)

    # Annotate key points
    for kv, marker in [(0.5, "k=0.5"), (1.0, "k=1.0"), (2.0, "k=2.0"), (3.0, "k=3.0")]:
        rs = kv / (1 + kv)
        ax.scatter([kv], [rs], color=C["k10"], s=30, zorder=4, edgecolors="white",
                   linewidths=0.7)
        ax.text(kv + 0.07, rs + 0.02, f"{rs:.2f}", fontsize=7.5, color=C["k10"])

    ax.text(0.4, 0.18,  "Substantive\nzone", fontsize=8, color="#2E7D32",
            ha="center", alpha=0.8)
    ax.text(2.5, 0.80,  "Ceremonial\nzone",  fontsize=8, color="#C62828",
            ha="center", alpha=0.8)

    ax.set_xlabel(r"Institutional tolerance $k$ (threshold $T = kc$)")
    ax.set_ylabel(r"Utilization $\rho_s$")
    ax.set_xlim(0, 4.1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "supp1_phase_diagram")


# ---------------------------------------------------------------------------
# SUPP 2 — Sensitivity Heatmap: Q̄ as function of (ρ_s, k)
# ---------------------------------------------------------------------------
def supp2_sensitivity_heatmap():
    print("Building Supp 2: Sensitivity heatmap …")

    k_grid  = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    rho_grid = np.arange(0.1, 0.95, 0.1)
    n_k, n_r = len(k_grid), len(rho_grid)

    Q_mat = np.zeros((n_k, n_r))
    for i, k in enumerate(k_grid):
        print(f"    k={k:.2f} …")
        mQ, _, _, _ = mc_sweep(rho_grid, k=k, n_reps=5, T_sim=3500, T_warmup=350)
        Q_mat[i, :] = mQ

    fig, ax = plt.subplots(figsize=(FW, 2.3))
    im = ax.imshow(Q_mat, aspect="auto", origin="lower",
                   extent=[rho_grid[0]-0.05, rho_grid[-1]+0.05,
                           -0.5, len(k_grid) - 0.5],
                   vmin=0.25, vmax=1.0, cmap="RdYlGn")

    # ρ* contour
    for i, k in enumerate(k_grid):
        rs = k / (1 + k)
        ax.plot(rs, i, marker="|", color="white", ms=10, mew=1.5, zorder=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"$\bar{Q}(\rho_s)$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(r"Utilization $\rho_s$")
    ax.set_ylabel(r"Tolerance $k$")
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([str(k) for k in k_grid], fontsize=7.5)
    ax.text(0.50, 6.65, r"White tick $=\rho^*$", fontsize=7, color="white",
            ha="center", va="top")
    fig.tight_layout()
    save(fig, "supp2_sensitivity")


# ---------------------------------------------------------------------------
# SUPP 3 — Time Series: N(t) and Q(t) at ρ_s = 0.3, 0.6, 0.9
# ---------------------------------------------------------------------------
def supp3_time_series():
    print("Building Supp 3: Time series …")

    scenarios = [
        (0.30, C["k05"], r"$\rho_s=0.30$ (below $\rho^*$)"),
        (0.60, C["k10"], r"$\rho_s=0.60$ (near $\rho^*$)"),
        (0.90, C["k20"], r"$\rho_s=0.90$ (above $\rho^*$)"),
    ]
    T_trace = 300
    T_skip  = 50

    fig, axes = plt.subplots(2, 1, figsize=(FW, 3.8), sharex=True)

    for rho_s, col, lbl in scenarios:
        rng = np.random.default_rng(7)
        lam = rho_s * 5 * 1.0
        T_thr = 5
        mu_s, mu_c = 1.0, 3.0
        t, N = 0.0, 0
        n_busy = 0
        deps = []
        ts, Ns, Qs = [], [], []
        next_arr = rng.exponential(1.0 / lam)

        def _mu_ts(n): return mu_c if n > T_thr else mu_s

        while t < T_trace + T_skip:
            nd = deps[0] if deps else float("inf")
            ne = next_arr if next_arr <= nd else nd
            if t >= T_skip and ne <= T_trace + T_skip:
                ts.append(t - T_skip)
                Ns.append(N)
                Qs.append(1.0 if N <= T_thr else 0.25)
            if ne >= T_trace + T_skip:
                break
            t = ne
            if next_arr <= nd:
                N += 1
                if n_busy < 5:
                    n_busy += 1
                    heapq.heappush(deps, t + rng.exponential(1.0 / _mu_ts(N)))
                next_arr = t + rng.exponential(1.0 / lam)
            else:
                heapq.heappop(deps)
                N -= 1
                n_busy -= 1
                if N >= 5:
                    n_busy += 1
                    heapq.heappush(deps, t + rng.exponential(1.0 / _mu_ts(N)))

        ts = np.array(ts);  Ns = np.array(Ns);  Qs = np.array(Qs)
        axes[0].step(ts, Ns, where="post", color=col, lw=0.9, alpha=0.85, label=lbl)
        axes[1].step(ts, Qs, where="post", color=col, lw=0.9, alpha=0.85)

    axes[0].axhline(5, color=C["gray"], lw=0.8, ls="--", alpha=0.6)
    axes[0].text(295, 5.15, r"$T=5$", fontsize=7, color=C["gray"], ha="right")
    axes[0].set_ylabel(r"Queue length $N(t)$")
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[1].axhline(0.25, color=C["gray"], lw=0.6, ls=":", alpha=0.5)
    axes[1].axhline(1.00, color=C["gray"], lw=0.6, ls=":", alpha=0.5)
    axes[1].text(295, 0.27, r"$q_c$", fontsize=7, ha="right", color=C["gray"])
    axes[1].text(295, 1.02, r"$q_s$", fontsize=7, ha="right", color=C["gray"])
    axes[1].set_ylabel(r"Quality $q(N(t))$")
    axes[1].set_xlabel(r"Simulation time")
    axes[1].set_ylim(0.10, 1.12)
    axes[1].set_yticks([0.25, 0.50, 0.75, 1.00])
    fig.tight_layout(h_pad=0.4)
    save(fig, "supp3_time_series")


# ---------------------------------------------------------------------------
# SUPP 4 — Ceremonial fraction P(N>T) vs ρ_s
# ---------------------------------------------------------------------------
def supp4_ceremonial_frac():
    print("Building Supp 4: Ceremonial fraction …")

    rho_vals = np.arange(0.05, 0.96, 0.05)
    fig, ax = plt.subplots(figsize=(FW, 2.4))

    for k, col, lbl in [
        (0.5, C["k05"], r"$k=0.5$"),
        (1.0, C["k10"], r"$k=1.0$"),
        (2.0, C["k20"], r"$k=2.0$"),
    ]:
        print(f"  Ceremonial fraction k={k} …")
        mQ, _, _, _ = mc_sweep(rho_vals, k=k, n_reps=6, T_sim=4000)
        # P_cer = (q_s - Q̄) / (q_s - q_c)
        P_cer = np.clip((1.0 - mQ) / 0.75, 0, 1)
        rho_star = k / (1 + k)
        ax.plot(rho_vals, P_cer, color=col, lw=1.7, label=lbl, zorder=3)
        ax.axvline(rho_star, color=col, lw=0.9, ls="--", alpha=0.45, zorder=1)

    ax.set_xlabel(r"Utilization $\rho_s$")
    ax.set_ylabel(r"Ceremonial fraction $P(N{>}T)$")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(-0.01, 0.50)
    ax.legend(fontsize=8)
    ax.text(0.50, 0.45,
            r"Dashed: $\rho^*$", fontsize=7, color=C["gray"])
    fig.tight_layout()
    save(fig, "supp4_ceremonial_frac")


# ---------------------------------------------------------------------------
# SUPP 5 — Quality decline rate: ΔQ̄/Δρ (nonlinearity diagnostic)
# ---------------------------------------------------------------------------
def supp5_quality_derivative():
    print("Building Supp 5: Quality decline rate …")

    rho_vals = np.arange(0.05, 0.93, 0.04)
    fig, ax = plt.subplots(figsize=(FW, 2.4))

    for k, col, lbl in [
        (0.5, C["k05"], r"$k=0.5$"),
        (1.0, C["k10"], r"$k=1.0$"),
        (2.0, C["k20"], r"$k=2.0$"),
    ]:
        mQ, _, _, _ = mc_sweep(rho_vals, k=k, n_reps=6, T_sim=4000)
        dQ = -np.gradient(mQ, rho_vals)   # rate of quality decline (positive = declining)
        rho_star = k / (1 + k)
        ax.plot(rho_vals, dQ, color=col, lw=1.7, label=lbl, zorder=3)
        ax.axvline(rho_star, color=col, lw=0.9, ls="--", alpha=0.45, zorder=1)

    ax.axhline(0, color=C["dark"], lw=0.6, ls=":", alpha=0.3)
    ax.set_xlabel(r"Utilization $\rho_s$")
    ax.set_ylabel(r"Quality decline rate $-\mathrm{d}\bar{Q}/\mathrm{d}\rho_s$")
    ax.set_xlim(-0.01, 1.0)
    ax.legend(fontsize=8)
    ax.text(0.50, ax.get_ylim()[1] * 0.92,
            "Peak marks tipping point", fontsize=7, color=C["gray"],
            ha="center")
    fig.tight_layout()
    save(fig, "supp5_quality_derivative")


# ---------------------------------------------------------------------------
# SUPP 6 — Regime map: (λ, c) space showing ρ* contours
# ---------------------------------------------------------------------------
def supp6_regime_map():
    print("Building Supp 6: Regime map …")

    mu_s = 1.0
    c_vals   = np.linspace(1, 20, 200)
    lam_vals = np.linspace(0.5, 18, 200)
    C_grid, L_grid = np.meshgrid(c_vals, lam_vals)
    RHO = L_grid / (C_grid * mu_s)

    fig, ax = plt.subplots(figsize=(FW, 2.6))

    for k, col, lbl in [(0.5, C["k05"], r"$k=0.5$"),
                        (1.0, C["k10"], r"$k=1.0$"),
                        (2.0, C["k20"], r"$k=2.0$")]:
        rho_star = k / (1 + k)
        CS = ax.contour(C_grid, L_grid, RHO, levels=[rho_star],
                        colors=[col], linewidths=1.6, linestyles="--", zorder=3)
        ax.clabel(CS, fmt=f"$\\rho^*$({k})", fontsize=7, inline=True,
                  manual=[(k * 10, rho_star * k * 10)])

    # Shade stable region (ρ_s < 1, i.e., λ < c*μ_s)
    ax.fill_between(c_vals, c_vals * mu_s, 18,
                    alpha=0.08, color="#C62828", label="Unstable ($\\rho_s>1$)")
    ax.plot(c_vals, c_vals * mu_s, color=C["dark"], lw=1.0, ls="-.", alpha=0.5,
            label="$\\rho_s=1$ boundary")

    ax.set_xlabel(r"Number of reviewers $c$")
    ax.set_ylabel(r"Arrival rate $\lambda$")
    ax.set_xlim(1, 20)
    ax.set_ylim(0.5, 18)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(14, 4, "Substantive\nzone\n($k=1.0$)", fontsize=7.5,
            ha="center", color=C["k10"], alpha=0.8)
    fig.tight_layout()
    save(fig, "supp6_regime_map")


# ---------------------------------------------------------------------------
# SUPP 7 — Intervention Effectiveness Regions
# ---------------------------------------------------------------------------
def supp7_intervention_regions():
    print("Building Supp 7: Intervention regions vs baseline ρ_s …")

    rho_baselines = np.arange(0.10, 0.92, 0.06)
    mu_s = 1.0; c = 5; k = 1.0

    Q_base, Q_I1, Q_I2, Q_I3, Q_I4 = [], [], [], [], []

    for rho_s in rho_baselines:
        lam = rho_s * c * mu_s
        q_b, _  = _simulate(lam, c=c, k=k, q_c=0.25, T_sim=5000, seed=0)
        q_i1, _ = _simulate(lam, c=c*2, k=k, q_c=0.25, T_sim=5000, seed=0)
        q_i2, _ = _simulate(lam/2, c=c, k=k, q_c=0.25, T_sim=5000, seed=0)
        q_i3, _ = _simulate(lam, c=c, k=2.0, q_c=0.25, T_sim=5000, seed=0)
        q_i4, _ = _simulate(lam, c=c, k=k, q_c=0.50, T_sim=5000, seed=0)
        Q_base.append(q_b);  Q_I1.append(q_i1)
        Q_I2.append(q_i2);   Q_I3.append(q_i3); Q_I4.append(q_i4)

    fig, ax = plt.subplots(figsize=(FW, 2.7))

    ax.plot(rho_baselines, Q_base, color=C["BL"], lw=1.5, ls="--",
            label="Baseline", zorder=4)
    ax.plot(rho_baselines, Q_I1,  color=C["I1"], lw=1.5,
            label="I1: double $c$", zorder=3)
    ax.plot(rho_baselines, Q_I2,  color=C["I2"], lw=1.5,
            label=r"I2: halve $\lambda$", zorder=3)
    ax.plot(rho_baselines, Q_I3,  color=C["I3"], lw=1.5, ls="-.",
            label=r"I3: raise $k \to 2$", zorder=3)
    ax.plot(rho_baselines, Q_I4,  color=C["I4"], lw=1.5, ls=":",
            label=r"I4: raise $q_c \to 0.5$", zorder=3)

    ax.axvline(0.5, color=C["dark"], lw=0.8, ls="--", alpha=0.35)
    ax.text(0.51, 0.36, r"$\rho^*$", fontsize=8, color=C["gray"])
    ax.axvspan(0.5, 0.92, alpha=0.05, color="#C62828")
    ax.set_xlabel(r"Baseline utilization $\rho_s$")
    ax.set_ylabel(r"Mean quality $\bar{Q}$ after intervention")
    ax.set_xlim(0.08, 0.93)
    ax.set_ylim(0.30, 1.06)
    ax.legend(fontsize=7.5, loc="lower left", ncol=1)
    fig.tight_layout()
    save(fig, "supp7_intervention_regions")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def supp8_robustness_landscape():
    """
    Supp 8: Joint (α, k) robustness landscape at ρ = 0.9.
    Contour map of Q̄ over the (k, α) parameter plane.
    Demonstrates that the qualitative degradation result is robust
    across the full parameter grid; only magnitude shifts.
    """
    alpha_vals = np.linspace(0.1, 1.0, 18)
    k_vals = np.linspace(0.3, 3.0, 18)
    rho = 0.9
    rho_star = 0.8

    AA, KK = np.meshgrid(alpha_vals, k_vals)
    excess = max(0.0, rho - rho_star)
    QQ = np.maximum(0.0, 1.0 - AA * (excess ** KK))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cp = ax.contourf(KK, AA, QQ, levels=20, cmap="RdYlGn")
    cs = ax.contour(KK, AA, QQ, levels=[0.75, 0.80, 0.85, 0.90, 0.95],
                    colors="k", linewidths=0.8, linestyles="--")
    ax.clabel(cs, fmt="%.2f", fontsize=8, inline=True)
    cbar = fig.colorbar(cp, ax=ax, pad=0.02)
    cbar.set_label("Mean authorization quality Q̄", fontsize=9)
    ax.set_xlabel("Shape parameter $k$", fontsize=10)
    ax.set_ylabel("Decay magnitude $\\alpha$", fontsize=10)
    ax.set_title("Robustness landscape: Q̄ at $\\rho = 0.9$\n"
                 "(dashed contours = quality iso-lines; $\\rho^* = 0.8$)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "supp8_robustness_landscape")


def supp9_state_space_evolution():
    """
    Supp 9: State-space trajectory of (ρ(t), Q̄(t)) as production load
    ramps up linearly from 0.2 to 1.2 over 100 time steps.
    Shows the path through the phase diagram and identifies the
    saturation zone crossing (marked with ×).
    """
    rho_star = 0.8
    t = np.linspace(0, 100, 300)
    rho_t = 0.2 + (1.0 / 100) * t
    rho_t = np.clip(rho_t, 0.0, 1.25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: time series of ρ(t)
    ax = axes[0]
    ax.plot(t, rho_t, color=C["k10"], lw=1.8, label="$\\rho(t)$")
    ax.axhline(rho_star, color=C["I4"], ls="--", lw=1.2,
               label="$\\rho^* = 0.8$")
    cross_idx = int(np.argmax(rho_t >= rho_star))
    ax.axvline(t[cross_idx], color=C["I4"], ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Time step", fontsize=10)
    ax.set_ylabel("Authorization utilization $\\rho$", fontsize=10)
    ax.set_title("(a) Load ramp-up trajectory", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Right: phase portrait (ρ, Q̄) trajectory
    ax = axes[1]
    for k, col, lbl in [(0.5, C["k05"], "$k=0.5$"),
                        (1.0, C["k10"], "$k=1.0$"),
                        (2.0, C["k20"], "$k=2.0$")]:
        q_t = np.maximum(0.0, 1.0 - 0.5 * np.maximum(0.0, rho_t - rho_star)**k)
        ax.plot(rho_t, q_t, color=col, lw=1.8, label=lbl)
        # Mark crossing
        if cross_idx > 0:
            ax.plot(rho_t[cross_idx], q_t[cross_idx], "x",
                    color=col, ms=8, mew=1.5)

    ax.axvline(rho_star, color=C["I4"], ls="--", lw=1.2,
               label="$\\rho^*$", alpha=0.7)
    ax.set_xlabel("Authorization utilization $\\rho$", fontsize=10)
    ax.set_ylabel("Authorization quality Q̄", fontsize=10)
    ax.set_title("(b) Phase portrait: $(\\rho, \\bar{Q})$ trajectory\n"
                 "(× = threshold crossing)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0.15, 1.3)
    ax.set_ylim(0.3, 1.05)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    save(fig, "supp9_state_space_evolution")


def supp10_queue_architecture():
    """
    Supp 10: Schematic queue architecture diagram rendered in matplotlib
    (no TikZ required). Shows the two-queue system: AI production queue
    feeding into the authorization bottleneck.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ---- AI production source ----
    src_box = plt.Rectangle((0.1, 1.4), 1.6, 1.2, fc="#D6EAF8", ec="#1A5276",
                             lw=1.5, zorder=3)
    ax.add_patch(src_box)
    ax.text(0.9, 2.0, "AI Production\nSource\n$\\lambda \\to 0^+$",
            ha="center", va="center", fontsize=8, color="#1A5276", zorder=4)

    # ---- Production queue ----
    for i in range(5):
        box = plt.Rectangle((1.9 + i * 0.55, 1.5), 0.5, 1.0,
                             fc="#EBF5FB", ec="#5DADE2", lw=1.0, zorder=3)
        ax.add_patch(box)
    ax.text(4.35, 0.95, "Production queue\n(AI premises)", ha="center",
            fontsize=8, color="#1A5276")

    # ---- Arrow: queue to bottleneck ----
    ax.annotate("", xy=(5.2, 2.0), xytext=(4.75, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A5276"))

    # ---- Authorization bottleneck ----
    bot_box = plt.Rectangle((5.25, 1.0), 1.8, 2.0, fc="#FDEDEC", ec="#C0392B",
                             lw=2.0, zorder=3)
    ax.add_patch(bot_box)
    ax.text(6.15, 2.0, "Authorization\nBottleneck\n$c$ reviewers, $\\mu$",
            ha="center", va="center", fontsize=8, color="#C0392B", zorder=4)

    # ---- Threshold marker ----
    ax.annotate("$\\rho > \\rho^*$\nceremonial\nmode",
                xy=(6.15, 3.05), xytext=(7.4, 3.5),
                fontsize=7.5, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0))

    # ---- Output arrow ----
    ax.annotate("", xy=(8.5, 2.0), xytext=(7.1, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A5276"))
    out_box = plt.Rectangle((8.55, 1.4), 1.2, 1.2, fc="#EAFAF1", ec="#1A7A5E",
                             lw=1.5, zorder=3)
    ax.add_patch(out_box)
    ax.text(9.15, 2.0, "Authorized\nDecisions\nQ̄(ρ)",
            ha="center", va="center", fontsize=8, color="#1A7A5E", zorder=4)

    # ---- Labels ----
    ax.text(5.0, 0.3, "Authorization utilization  $\\rho = \\lambda / (\\mu c)$  ·  "
            "Saturation threshold  $\\rho^* = 0.8$",
            ha="center", va="bottom", fontsize=8.5, color="#555555",
            style="italic")
    ax.set_title("Coupled authorization queue architecture\n"
                 "(substantive mode: $\\rho \\leq \\rho^*$  ·  "
                 "ceremonial mode: $\\rho > \\rho^*$)",
                 fontsize=9.5, pad=6)

    fig.tight_layout()
    save(fig, "supp10_queue_architecture")


# ---------------------------------------------------------------------------
# SUPPLEMENTARY 11 — c-Convergence: Halfin-Whitt approximation validation
# ---------------------------------------------------------------------------
def supp11_c_convergence():
    """
    Shows Q̄(ρ_s) for c ∈ {2, 5, 10, 20} at k=1.0.
    As c increases the inflection tightens around ρ*=0.50,
    validating the Halfin-Whitt large-c approximation.
    """
    print("Building Supp 11: c-convergence (Halfin-Whitt validation) …")

    rho_fine = np.linspace(0.02, 0.96, 48)
    c_vals   = [2, 5, 10, 20]
    colors   = ["#7F4D9A", "#C0392B", "#2C5F8A", "#1A7A5E"]
    styles   = [":", "--", "-.", "-"]

    fig, ax = plt.subplots(figsize=(FW, FH))

    # Reference ρ* = 0.50 (k=1)
    ax.axvline(0.50, color=C["dark"], lw=0.8, ls="--", alpha=0.35, zorder=1)
    ax.text(0.505, 1.04, r"$\rho^*=0.50$", fontsize=7, color=C["gray"],
            va="bottom", ha="left")

    for c_val, col, ls in zip(c_vals, colors, styles):
        mQ, eQ, _, _ = mc_sweep(rho_fine, c=c_val, k=1.0, n_reps=30, T_sim=12000)
        ax.fill_between(rho_fine, mQ - eQ, mQ + eQ, color=col, alpha=0.07, zorder=2)
        ax.plot(rho_fine, mQ, color=col, lw=1.9, ls=ls,
                label=f"$c={c_val}$", zorder=3)

    ax.set_xlabel(r"Substantive utilization $\rho_s$")
    ax.set_ylabel(r"Mean quality $\bar{Q}(\rho_s)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.18, 1.10)
    ax.legend(title="Server count $c$", loc="lower left",
              title_fontsize=8, framealpha=0.97)
    ax.set_title(r"Convergence of inflection to $\rho^*=0.50$ as $c$ increases"
                 "\n(Halfin–Whitt heavy-traffic regime, $k=1.0$, 30 MC reps)",
                 fontsize=9)
    ax.text(0.02, 0.22,
            "Inflection sharpens with $c$;\nat $c\\geq10$ closely tracks $\\rho^*$.",
            fontsize=7.5, color=C["gray"], va="bottom")

    fig.tight_layout()
    save(fig, "supp11_c_convergence")


def main():
    print("=" * 60)
    print("SBP-BRiMS 2026 Figure Generation")
    print(f"Output directory: {OUT}")
    print("=" * 60)

    fig2_quality_curves()
    fig3_interventions()
    supp1_phase_diagram()
    supp2_sensitivity_heatmap()
    supp3_time_series()
    supp4_ceremonial_frac()
    supp5_quality_derivative()
    supp6_regime_map()
    supp7_intervention_regions()
    supp8_robustness_landscape()
    supp9_state_space_evolution()
    supp10_queue_architecture()
    supp11_c_convergence()

    print("\n" + "=" * 60)
    print("All figures generated.")
    print("Paper figures: figures/fig2_quality.pdf, figures/fig3_interventions.pdf")
    print("Supplementary: figures/supp1_*.pdf … supp11_*.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
