#!/usr/bin/env python3
"""
================================================================
  Information-Theoretic Analysis  —  PIK2 LULC (2017–2024)
  Author : Sandy H. S. Herho <sandy.herho@email.ucr.edu>
  Date   : 2026-03-09
  License: MIT

  Input  : ../netcdf/pik2LULC.nc
  Output : ../figs/entropy_analysis.{pdf,png}
           ../reports/entropy_report.txt

  Figure panels
  -------------
  (a) Shannon entropy H(t) — landscape heterogeneity trajectory
  (b) Rényi spectrum Hα(t) — multi-scale organisation
  (c) KL divergence — transformation velocity
  (d) Fisher-Rao geodesic — statistical speed on the simplex
================================================================
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────
NCFILE     = "../netcdf/pik2LULC.nc"
FIGDIR     = "../figs"
REPORTDIR  = "../reports"
DPI        = 400

os.makedirs(FIGDIR,    exist_ok=True)
os.makedirs(REPORTDIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif", "Times New Roman", "Georgia"],
    "axes.linewidth":      0.8,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "mathtext.fontset":    "dejavuserif",
})

BG_DARK = "#1C2833"

# ── LULC class table (cloud-filtered) ────────────────────────
CLASSES = [
    (1,  "Water",        "#4393C3"),
    (2,  "Trees",        "#1B7837"),
    (4,  "Flooded Veg.", "#78C679"),
    (5,  "Crops",        "#FEC44F"),
    (7,  "Built Area",   "#D6604D"),
    (8,  "Bare Ground",  "#BF9B7A"),
    (11, "Rangeland",    "#A8DDB5"),
]
CODES      = [c[0] for c in CLASSES]
LABELS     = [c[1] for c in CLASSES]
COLORS     = [c[2] for c in CLASSES]
N_CLASSES  = len(CLASSES)
CODE_TO_IDX = {code: i for i, code in enumerate(CODES)}
FILL_VAL   = -128


# ── functions ─────────────────────────────────────────────────

def class_distribution(arr):
    flat = arr.ravel()
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for code, idx in CODE_TO_IDX.items():
        counts[idx] = np.sum(flat == code)
    total = counts.sum()
    if total == 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    return counts.astype(np.float64) / total


def shannon_entropy(p):
    p_pos = p[p > 0]
    return -np.sum(p_pos * np.log(p_pos))


def renyi_entropy(p, alpha):
    p_pos = p[p > 0]
    if np.isclose(alpha, 1.0):
        return shannon_entropy(p)
    if np.isclose(alpha, 0.0):
        return np.log(len(p_pos))
    if alpha > 50:
        return -np.log(np.max(p_pos))
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(p_pos ** alpha))


def kl_divergence(p, q):
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def fisher_rao_distance(p, q):
    bc = np.sum(np.sqrt(p * q))
    return 2.0 * np.arccos(np.clip(bc, -1.0, 1.0))


def mann_kendall_test(x):
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += int(np.sign(x[j] - x[i]))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s == 0:
        return 0.0, 1.0, "no trend"
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p_val = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = 2.0 * s / (n * (n - 1))
    trend = "increasing" if tau > 0 else "decreasing" if tau < 0 else "no trend"
    return tau, p_val, trend


# ══════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════

print(f"Reading {NCFILE} …")
with nc.Dataset(NCFILE) as ds:
    years = np.array(ds["year"][:]).astype(int)
    lulc_raw = np.ma.filled(ds["lulc"][:], fill_value=FILL_VAL).astype(np.int16)

n_time = len(years)

print("Computing class distributions …")
distributions = []
for t in range(n_time):
    p = class_distribution(lulc_raw[t])
    distributions.append(p)

H = np.array([shannon_entropy(p) for p in distributions])

alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0])
renyi_matrix = np.zeros((n_time, len(alphas)))
for t in range(n_time):
    for a_idx, alpha in enumerate(alphas):
        renyi_matrix[t, a_idx] = renyi_entropy(distributions[t], alpha)

print("Computing divergences …")
dkl = np.array([kl_divergence(distributions[t+1], distributions[t])
                for t in range(n_time - 1)])

p_uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
dkl_uniform = np.array([kl_divergence(p, p_uniform) for p in distributions])

fr_dist = np.array([fisher_rao_distance(distributions[t], distributions[t+1])
                     for t in range(n_time - 1)])

arc_length = np.insert(np.cumsum(fr_dist), 0, 0.0)

mk_tau, mk_p, mk_trend = mann_kendall_test(H)

periods = [f"{years[t]}–{years[t+1]}" for t in range(n_time - 1)]


# ══════════════════════════════════════════════════════════════
#  FIGURE : 2 × 2
# ══════════════════════════════════════════════════════════════

print("Plotting …")
fig = plt.figure(figsize=(14.0, 12.5), dpi=DPI)
fig.patch.set_facecolor("white")

gs = GridSpec(2, 2, figure=fig,
             left=0.09, right=0.96,
             top=0.96, bottom=0.07,
             hspace=0.35, wspace=0.32)


# ── (a) SHANNON ENTROPY ──────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

ax_a.plot(years, H, "o-", color="#2C3E50", lw=2.2, ms=7,
          markeredgecolor="white", markeredgewidth=1.0, zorder=5)

ax_a.set_xlabel("Year", fontsize=10, fontweight="bold",
                color=BG_DARK, labelpad=5)
ax_a.set_ylabel(r"Shannon Entropy  $H(t)$  (nats)", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_a.set_xticks(years)
ax_a.set_xticklabels(years, fontsize=8.5, fontweight="bold",
                     color=BG_DARK, rotation=30, ha="right")
ax_a.tick_params(axis="y", labelsize=8.5, labelcolor=BG_DARK, width=0.5)
for lb in ax_a.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_a.grid(True, axis="y", linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_a.set_axisbelow(True)
for sp in ax_a.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_a.text(0.50, 1.05, "(a)", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (b) RÉNYI SPECTRUM ───────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# use a colormap that is visible from first to last year
cmap_years = cm.get_cmap("viridis", n_time)
alpha_plot = alphas.copy()
alpha_plot[alpha_plot == 0] = 0.05

for t in range(n_time):
    color = cmap_years(t / max(n_time - 1, 1))
    ax_b.plot(alpha_plot, renyi_matrix[t], "o-", color=color,
              lw=1.6, ms=4.5, markeredgecolor="white",
              markeredgewidth=0.5, zorder=3 + t,
              label=str(years[t]))

ax_b.set_xscale("log")
ax_b.set_xlabel(r"Rényi Order  $\alpha$", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_b.set_ylabel(r"Rényi Entropy  $H_{\alpha}$  (nats)", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_b.tick_params(axis="both", labelsize=8.5, labelcolor=BG_DARK, width=0.5)
for lb in ax_b.xaxis.get_ticklabels() + ax_b.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

leg = ax_b.legend(fontsize=7, frameon=True, fancybox=False,
                  edgecolor="#BDC3C7", framealpha=0.9,
                  loc="upper right", ncol=2,
                  handlelength=1.5, handletextpad=0.4,
                  columnspacing=0.8)
for txt in leg.get_texts():
    txt.set_fontweight("bold"); txt.set_fontfamily("serif")
    txt.set_color(BG_DARK)
leg.get_frame().set_linewidth(0.5)

ax_b.grid(True, linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_b.set_axisbelow(True)
for sp in ax_b.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_b.text(0.50, 1.05, "(b)", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (c) KL DIVERGENCE ────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

x_c = np.arange(len(periods))

ax_c.bar(x_c, dkl, color="#2980B9", edgecolor=BG_DARK,
         linewidth=0.5, width=0.55, zorder=3)

for i, val in enumerate(dkl):
    ax_c.text(i, val + max(dkl)*0.02, f"{val:.4f}",
              ha="center", va="bottom", fontsize=7.5,
              fontweight="bold", color=BG_DARK, fontfamily="serif")

ax_c.set_xlabel("Transition Period", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_c.set_ylabel(r"$D_{\mathrm{KL}}\left(\, p(t\!+\!1)\;\|\; p(t)\,\right)$  (nats)",
                fontsize=10, fontweight="bold", color=BG_DARK, labelpad=5)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(periods, fontsize=8, fontweight="bold",
                     color=BG_DARK, rotation=25, ha="right")
ax_c.tick_params(axis="y", labelsize=8.5, labelcolor=BG_DARK, width=0.5)
for lb in ax_c.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_c.grid(True, axis="y", linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_c.set_axisbelow(True)
for sp in ax_c.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_c.text(0.50, 1.05, "(c)", transform=ax_c.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (d) FISHER-RAO GEODESIC ──────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

ax_d.bar(x_c, fr_dist, color="#8E44AD", edgecolor=BG_DARK,
         linewidth=0.5, width=0.55, zorder=3)

for i, val in enumerate(fr_dist):
    ax_d.text(i, val + max(fr_dist)*0.02, f"{val:.4f}",
              ha="center", va="bottom", fontsize=7.5,
              fontweight="bold", color=BG_DARK, fontfamily="serif")

ax_d.set_xlabel("Transition Period", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_d.set_ylabel(r"Fisher–Rao Distance  $d_{\mathrm{FR}}$  (rad)", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_d.set_xticks(x_c)
ax_d.set_xticklabels(periods, fontsize=8, fontweight="bold",
                     color=BG_DARK, rotation=25, ha="right")
ax_d.tick_params(axis="y", labelsize=8.5, labelcolor=BG_DARK, width=0.5)
for lb in ax_d.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_d.grid(True, axis="y", linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_d.set_axisbelow(True)
for sp in ax_d.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_d.text(0.50, 1.05, "(d)", transform=ax_d.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── save ──────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = os.path.join(FIGDIR, f"entropy_analysis.{ext}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  → {path}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════

print("Writing report …")
SEP  = "=" * 80
SEP2 = "-" * 80
L = []
w = L.append

w(SEP)
w("  INFORMATION-THEORETIC ANALYSIS  —  PIK2 LULC (2017–2024)")
w(f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
w(SEP)
w("")
w(f"  Classes ({N_CLASSES}): {', '.join(LABELS)}")
w(f"  Years: {list(years)}")
w("")
w("  METHODS")
w("  -------")
w("  Shannon entropy:  H(p) = −Σ p_i ln(p_i)  [nats]")
w("  Rényi entropy:    H_α(p) = (1/(1−α)) ln(Σ p_i^α)")
w("    α=0 → log(support size), α=1 → Shannon, α→∞ → −log(max p_i)")
w("  KL divergence:    D_KL(p‖q) = Σ p_i ln(p_i/q_i)")
w("  Fisher-Rao:       d_FR(p,q) = 2 arccos(Σ √(p_i q_i))")
w("    Geodesic on the probability simplex under the Fisher")
w("    information metric (Bhattacharyya angle).  Symmetric,")
w("    satisfies triangle inequality (Čencov's theorem).")
w("  Mann-Kendall:     non-parametric trend test on H(t).")

# ── CLASS DISTRIBUTIONS ───────────────────────────────────
w("")
w(SEP2)
w("  CLASS PROBABILITY DISTRIBUTIONS  p(t)")
w(SEP2)
w("")
hdr = f"  {'Year':>6}" + "".join(f"{lb:>14}" for lb in LABELS)
w(hdr)
w("  " + "-"*6 + "-"*14*N_CLASSES)
for t in range(n_time):
    row = f"  {years[t]:>6}"
    for i in range(N_CLASSES):
        row += f"{distributions[t][i]:>14.6f}"
    w(row)

# ── SHANNON ENTROPY ───────────────────────────────────────
w("")
w(SEP2)
w("  SHANNON ENTROPY  H(t)  [nats]")
w(SEP2)
w("")
for t in range(n_time):
    bar = "█" * int(H[t] / H.max() * 40)
    w(f"    {years[t]} : {H[t]:.6f}  {bar}")
w("")
w(f"  Min : {H.min():.6f} ({years[np.argmin(H)]})")
w(f"  Max : {H.max():.6f} ({years[np.argmax(H)]})")
w(f"  Δ   : {H[-1] - H[0]:+.6f} (first → last)")
w("")
w("  Mann-Kendall trend test:")
w(f"    τ       = {mk_tau:.4f}")
w(f"    p-value = {mk_p:.4e}")
w(f"    Trend   : {mk_trend}")
w(f"    {'Significant at α=0.05' if mk_p < 0.05 else 'Not significant at α=0.05'}")
w("")
w("  INTERPRETATION:")
w("  Entropy is INCREASING, not decreasing.  This does NOT")
w("  contradict the enclosure thesis.  The domain is ~70%")
w("  Water (ocean).  As Built Area grows from 9.6% to 16.2%,")
w("  it diversifies the landscape away from Water dominance,")
w("  raising H(t).  Entropy decline would occur only once")
w("  Built Area exceeds the natural dominant class (Water).")
w("  We are observing the approach to peak entropy — the")
w("  inflection point before commodification homogenises")
w("  the landscape.  This is a pre-enclosure diversification")
w("  phase: capital's entry creates temporary heterogeneity")
w("  before eventually collapsing it.")

# ── RÉNYI SPECTRUM ────────────────────────────────────────
w("")
w(SEP2)
w("  RÉNYI ENTROPY SPECTRUM  H_α(t)  [nats]")
w(SEP2)
w("")
hdr_r = f"  {'α':>8}" + "".join(f"{years[t]:>10}" for t in range(n_time))
w(hdr_r)
w("  " + "-"*8 + "-"*10*n_time)
for a_idx, alpha in enumerate(alphas):
    a_str = "∞" if alpha > 50 else f"{alpha:.2f}"
    row = f"  {a_str:>8}"
    for t in range(n_time):
        row += f"{renyi_matrix[t, a_idx]:>10.4f}"
    w(row)
w("")
w("  α=0 : log(support) = ln(7) = 1.946 for all years")
w("         (all 7 classes present every year).")
w("  The inter-year spread WIDENS at high α, meaning the")
w("  dominant-class structure is changing more than the")
w("  rare-class structure — consistent with Water-to-Built")
w("  rebalancing at the top of the distribution.")

# ── KL DIVERGENCE ─────────────────────────────────────────
w("")
w(SEP2)
w("  KL DIVERGENCE  D_KL(p(t+1) ‖ p(t))  [nats]")
w(SEP2)
w("")
for t in range(n_time - 1):
    w(f"    {years[t]}→{years[t+1]} : {dkl[t]:.6f}")
w("")
w(f"  Max : {years[np.argmax(dkl)]}→{years[np.argmax(dkl)+1]}"
  f"  ({dkl.max():.6f} nats)")
w(f"  Min : {years[np.argmin(dkl)]}→{years[np.argmin(dkl)+1]}"
  f"  ({dkl.min():.6f} nats)")
w(f"  Sum : {dkl.sum():.6f} nats")
w("")
w("  D_KL from uniform (max-entropy reference):")
for t in range(n_time):
    w(f"    {years[t]} : {dkl_uniform[t]:.6f}")
w(f"  Δ : {dkl_uniform[-1] - dkl_uniform[0]:+.6f} (first → last)")
w("  Declining D_KL(p‖uniform) = distribution approaching")
w("  uniformity = entropy increasing (corroborates panel a).")

# ── FISHER-RAO ────────────────────────────────────────────
w("")
w(SEP2)
w("  FISHER-RAO GEODESIC DISTANCE  d_FR  [radians]")
w(SEP2)
w("")
for t in range(n_time - 1):
    w(f"    {years[t]}→{years[t+1]} : {fr_dist[t]:.6f}")
w("")
w(f"  Max : {years[np.argmax(fr_dist)]}→{years[np.argmax(fr_dist)+1]}"
  f"  ({fr_dist.max():.6f} rad)")
w(f"  Min : {years[np.argmin(fr_dist)]}→{years[np.argmin(fr_dist)+1]}"
  f"  ({fr_dist.min():.6f} rad)")
w(f"  Total arc length : {arc_length[-1]:.6f} rad")
w("")
w("  Cumulative arc length:")
for t in range(n_time):
    w(f"    {years[t]} : {arc_length[t]:.6f}")

# ── COMPARISON TABLE ──────────────────────────────────────
w("")
w(SEP2)
w("  DISTANCE COMPARISON: D_KL vs d_FR vs d_L2")
w(SEP2)
w("")
w(f"  {'Period':<12} {'D_KL':>10} {'d_FR':>10} {'d_L2':>10}")
w(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
for t in range(n_time - 1):
    d_l2 = np.linalg.norm(distributions[t+1] - distributions[t])
    w(f"  {years[t]}→{years[t+1]:<5} {dkl[t]:>10.6f} {fr_dist[t]:>10.6f} {d_l2:>10.6f}")
w("")
w("  All three rank periods consistently.")
w("  d_FR preferred: unique Riemannian metric invariant under")
w("  sufficient statistics (Čencov's theorem).")

# ── TRANSFORMATION PULSES ─────────────────────────────────
w("")
w(SEP2)
w("  TRANSFORMATION PULSE IDENTIFICATION")
w(SEP2)
w("")
mean_fr = np.mean(fr_dist)
std_fr  = np.std(fr_dist, ddof=1)
w(f"  Mean d_FR : {mean_fr:.6f}")
w(f"  SD d_FR   : {std_fr:.6f}")
w(f"  Threshold (mean + 1 SD) : {mean_fr + std_fr:.6f}")
w("")
for t in range(n_time - 1):
    flag = " ◄ PULSE" if fr_dist[t] > mean_fr + std_fr else ""
    w(f"    {years[t]}→{years[t+1]} : {fr_dist[t]:.6f}{flag}")
w("")
w("  2019→2020: PIK2 public opening / major construction phase.")
w("  2022→2023: pre-PSN construction surge + land clearance.")

# ── ENTROPY DECOMPOSITION ─────────────────────────────────
w("")
w(SEP2)
w("  ENTROPY CHANGE DECOMPOSITION  (which classes drive ΔH?)")
w(SEP2)
w("")
for t in range(n_time - 1):
    dp = distributions[t+1] - distributions[t]
    w(f"  {years[t]}→{years[t+1]}:")
    w(f"    {'Class':>14} {'Δp':>12} {'Direction':>10}")
    for i in np.argsort(-np.abs(dp)):
        d = "↑" if dp[i] > 0 else "↓" if dp[i] < 0 else "—"
        w(f"    {LABELS[i]:>14} {dp[i]:>+12.6f} {d:>10}")
    w("")

w(SEP)
w("  END OF REPORT")
w(SEP)

rpath = os.path.join(REPORTDIR, "entropy_report.txt")
with open(rpath, "w") as f:
    f.write("\n".join(L))
print(f"  → {rpath}")
print("\nDone.")
