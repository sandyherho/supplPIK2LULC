#!/usr/bin/env python3
"""
================================================================
  Markov Transition Analysis  —  PIK2 LULC (2017–2024)
  Author : Sandy H. S. Herho <sandy.herho@email.ucr.edu>
  Date   : 2026-03-09
  License: MIT

  Input  : ../netcdf/pik2LULC.nc
  Output : ../figs/markov_analysis.{pdf,png}
           ../reports/markov_report.txt

  Figure panels
  -------------
  (a) Pooled transition matrix — structure of accumulation
  (b) Expected absorption time — speed of enclosure
  (c) Stationary distribution — predicted end-state
  (d) Built self-retention — irreversibility diagnostic
================================================================
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
})

BG_DARK = "#1C2833"

# ── LULC class table ─────────────────────────────────────────
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
BUILT_IDX  = CODE_TO_IDX[7]

TRANSIENT_IDX    = [i for i in range(N_CLASSES) if i != BUILT_IDX]
TRANSIENT_LABELS = [LABELS[i] for i in TRANSIENT_IDX]
TRANSIENT_COLORS = [COLORS[i] for i in TRANSIENT_IDX]
N_TRANSIENT      = len(TRANSIENT_IDX)


# ── core functions ────────────────────────────────────────────

def load_data():
    with nc.Dataset(NCFILE) as ds:
        years = np.array(ds["year"][:]).astype(int)
        lulc  = np.ma.filled(ds["lulc"][:], fill_value=FILL_VAL).astype(np.int16)
    return lulc, years


def count_transitions(arr_from, arr_to):
    ff, ft = arr_from.ravel(), arr_to.ravel()
    max_code = max(CODES) + 1
    lut = np.full(max_code + 1, -1, dtype=np.int8)
    for code, idx in CODE_TO_IDX.items():
        lut[code] = idx
    valid = (ff >= 0) & (ff <= max_code) & (ft >= 0) & (ft <= max_code)
    idx_f, idx_t = lut[ff[valid]], lut[ft[valid]]
    both = (idx_f >= 0) & (idx_t >= 0)
    idx_f, idx_t = idx_f[both], idx_t[both]
    flat = idx_f.astype(np.int64) * N_CLASSES + idx_t.astype(np.int64)
    return np.bincount(flat, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def counts_to_prob(C):
    rs = C.sum(axis=1, keepdims=True).astype(np.float64)
    rs[rs == 0] = 1.0
    return C.astype(np.float64) / rs


def asymptotic_se(P, C):
    rs = C.sum(axis=1).astype(np.float64)
    SE = np.zeros_like(P)
    for i in range(P.shape[0]):
        if rs[i] > 0:
            SE[i] = np.sqrt(P[i] * (1.0 - P[i]) / rs[i])
    return SE


def spectral_analysis(P):
    ev = np.linalg.eigvals(P)
    ev = ev[np.argsort(-np.abs(ev))]
    lam2 = np.abs(ev[1]) if len(ev) > 1 else 0.0
    return ev, 1.0 - lam2, (-1.0/np.log(lam2) if 0 < lam2 < 1 else np.inf)


def stationary_dist(P):
    ev, evec = np.linalg.eig(P.T)
    pi = np.abs(np.real(evec[:, np.argmin(np.abs(ev - 1.0))]))
    return pi / pi.sum()


def absorbing_chain_analysis(P, absorbing_idx):
    """
    Absorbing Markov chain with `absorbing_idx` as absorbing state.
    Q  = transient sub-matrix
    N  = (I − Q)⁻¹   fundamental matrix
    t  = N·1          expected absorption time (row sums of N)
    var = (2N−I)·t − t²   variance of absorption time
    """
    tidx = [i for i in range(P.shape[0]) if i != absorbing_idx]
    Q = P[np.ix_(tidx, tidx)]
    I = np.eye(len(tidx))
    N = np.linalg.inv(I - Q)
    t = N.sum(axis=1)
    var = ((2.0 * N - I) @ t.reshape(-1, 1)).ravel() - t**2
    var = np.maximum(var, 0.0)
    return t, np.sqrt(var), N, Q


def gtest_stationarity(count_matrices):
    C_pool = sum(count_matrices)
    P_pool = counts_to_prob(C_pool)
    G = 0.0
    for C_t in count_matrices:
        P_t = counts_to_prob(C_t)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if C_t[i, j] > 0 and P_pool[i, j] > 0:
                    G += 2.0 * C_t[i, j] * np.log(P_t[i, j] / P_pool[i, j])
    active = sum(1 for i in range(N_CLASSES)
                 if any(Ct[i].sum() > 0 for Ct in count_matrices))
    df = (len(count_matrices) - 1) * active * (N_CLASSES - 1)
    return G, df, (1.0 - chi2.cdf(G, df) if df > 0 else 1.0)


# ══════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════

print(f"Reading {NCFILE} …")
lulc, years = load_data()
n_time = len(years)

print("Computing transition matrices …")
count_mats, prob_mats, se_mats = [], [], []
gaps, mix_times, eigval_list = [], [], []

for t in range(n_time - 1):
    print(f"  {years[t]}–{years[t+1]} …")
    C  = count_transitions(lulc[t], lulc[t + 1])
    P  = counts_to_prob(C)
    SE = asymptotic_se(P, C)
    ev, gap, mt = spectral_analysis(P)
    count_mats.append(C); prob_mats.append(P); se_mats.append(SE)
    gaps.append(gap); mix_times.append(mt); eigval_list.append(ev)

periods = [f"{years[t]}–{years[t+1]}" for t in range(n_time - 1)]

C_pooled  = sum(count_mats)
P_pooled  = counts_to_prob(C_pooled)
SE_pooled = asymptotic_se(P_pooled, C_pooled)
ev_pooled, gap_pooled, mt_pooled = spectral_analysis(P_pooled)
pi_pooled = stationary_dist(P_pooled)

print("Absorbing chain analysis …")
abs_t, abs_sd, abs_N, Q_pooled = absorbing_chain_analysis(P_pooled, BUILT_IDX)

abs_t_per_period = []
for t in range(len(periods)):
    try:
        at, _, _, _ = absorbing_chain_analysis(prob_mats[t], BUILT_IDX)
        abs_t_per_period.append(at)
    except np.linalg.LinAlgError:
        abs_t_per_period.append(np.full(N_TRANSIENT, np.nan))

self_ret    = [prob_mats[t][BUILT_IDX, BUILT_IDX] for t in range(len(periods))]
self_ret_se = [se_mats[t][BUILT_IDX, BUILT_IDX] for t in range(len(periods))]

G_stat, G_df, G_pval = gtest_stationarity(count_mats)
frob_dists = [np.linalg.norm(prob_mats[t] - P_pooled, ord="fro")
              for t in range(len(periods))]


# ══════════════════════════════════════════════════════════════
#  FIGURE : 2 × 2
# ══════════════════════════════════════════════════════════════

print("Plotting …")
fig = plt.figure(figsize=(14.0, 12.5), dpi=DPI)
fig.patch.set_facecolor("white")

gs = GridSpec(2, 2, figure=fig,
             left=0.08, right=0.96,
             top=0.96, bottom=0.06,
             hspace=0.35, wspace=0.30)


# ── (a) POOLED TRANSITION HEATMAP ────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

im = ax_a.imshow(P_pooled, cmap="cividis", vmin=0, vmax=1,
                 aspect="equal", interpolation="nearest")

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        val = P_pooled[i, j]
        color = "white" if val < 0.5 else "black"
        ax_a.text(j, i, f"{val:.3f}", ha="center", va="center",
                  fontsize=7.0, fontweight="bold", color=color,
                  fontfamily="serif")

ax_a.set_xticks(range(N_CLASSES))
ax_a.set_yticks(range(N_CLASSES))
ax_a.set_xticklabels(LABELS, fontsize=7.5, fontweight="bold",
                     color=BG_DARK, rotation=40, ha="right")
ax_a.set_yticklabels(LABELS, fontsize=7.5, fontweight="bold",
                     color=BG_DARK)
ax_a.set_xlabel("Destination Class", fontsize=9.5,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_a.set_ylabel("Origin Class", fontsize=9.5,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_a.tick_params(axis="both", color=BG_DARK, length=0)

cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cbar.set_label("Transition Probability", fontsize=8.5,
               fontweight="bold", color=BG_DARK, labelpad=6)
cbar.ax.tick_params(labelsize=7.5, labelcolor=BG_DARK, width=0.5, length=2.5)
for lb in cbar.ax.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")
cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor(BG_DARK)
for sp in ax_a.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_a.text(0.50, 1.05, "(a)", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (b) EXPECTED ABSORPTION TIME — clean bars ────────────
ax_b = fig.add_subplot(gs[0, 1])

y_pos = np.arange(N_TRANSIENT)
bars_b = ax_b.barh(y_pos, abs_t, color=TRANSIENT_COLORS,
                   edgecolor=BG_DARK, linewidth=0.5,
                   height=0.55, zorder=3)

# annotate with E[T] value only
for i, val in enumerate(abs_t):
    ax_b.text(val + 1.5, i, f"{val:.1f} yr",
              va="center", ha="left", fontsize=8,
              fontweight="bold", color=BG_DARK, fontfamily="serif")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(TRANSIENT_LABELS, fontsize=8, fontweight="bold",
                     color=BG_DARK)
ax_b.set_xlabel("Expected Years to Absorption into Built Area",
                fontsize=9, fontweight="bold", color=BG_DARK, labelpad=5)
ax_b.tick_params(axis="x", labelsize=8, labelcolor=BG_DARK, width=0.5)
ax_b.tick_params(axis="y", width=0.5, length=0)
for lb in ax_b.xaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_b.invert_yaxis()
ax_b.set_xlim(0, max(abs_t) * 1.35)
ax_b.grid(True, axis="x", linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_b.set_axisbelow(True)
for sp in ax_b.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_b.text(0.50, 1.05, "(b)", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (c) STATIONARY DISTRIBUTION π ────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

x_c = np.arange(N_CLASSES)
bar_c = ax_c.bar(x_c, pi_pooled, color=COLORS, edgecolor=BG_DARK,
                 linewidth=0.5, width=0.6, zorder=3)

for bar, val in zip(bar_c, pi_pooled):
    ax_c.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.008,
              f"{val:.3f}", ha="center", va="bottom",
              fontsize=8, fontweight="bold", color=BG_DARK,
              fontfamily="serif")

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(LABELS, fontsize=7.5, fontweight="bold",
                     color=BG_DARK, rotation=40, ha="right")
ax_c.set_ylabel("Stationary Probability  π", fontsize=9.5,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_c.tick_params(axis="y", labelsize=8, labelcolor=BG_DARK, width=0.5)
ax_c.tick_params(axis="x", width=0.5)
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


# ── (d) BUILT SELF-RETENTION OVER TIME ───────────────────
ax_d = fig.add_subplot(gs[1, 1])

x_d = np.arange(len(periods))
sr  = np.array(self_ret)
se  = np.array(self_ret_se)

ax_d.fill_between(x_d, sr - 1.96*se, sr + 1.96*se,
                  color="#D6604D", alpha=0.15, zorder=2)
ax_d.plot(x_d, sr, "o-", color="#D6604D", lw=2.0, ms=6.5,
          markeredgecolor="white", markeredgewidth=1.0, zorder=5)
ax_d.axhline(1.0, color="#7F8C8D", lw=0.8, ls="--", alpha=0.6, zorder=1)

for i, val in enumerate(sr):
    leak = (1.0 - val) * 100
    ax_d.text(i, val - 0.003, f"{leak:.1f}%",
              ha="center", va="top", fontsize=7, fontweight="bold",
              color="#922B21", fontfamily="serif")

ax_d.set_xlabel("Transition Period", fontsize=9.5,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_d.set_ylabel("P(Built → Built)", fontsize=9.5,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_d.set_xticks(x_d)
ax_d.set_xticklabels(periods, fontsize=7.5, fontweight="bold",
                     color=BG_DARK, rotation=25, ha="right")
ax_d.tick_params(axis="y", labelsize=8, labelcolor=BG_DARK, width=0.5)
for lb in ax_d.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_d.set_ylim(min(sr) - 0.015, 1.005)
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
    path = os.path.join(FIGDIR, f"markov_analysis.{ext}")
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
w("  MARKOV TRANSITION ANALYSIS  —  PIK2 LULC (2017–2024)")
w(f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
w(SEP)
w("")
w(f"  Classes ({N_CLASSES}): {', '.join(LABELS)}")
w(f"  Periods: {len(periods)}   Years: {list(years)}")
w("")
w("  METHODS")
w("  -------")
w("  Transition probabilities: empirical pixel-wise counts,")
w("    row-normalised.  SE: SE(p_ij) = sqrt(p_ij(1-p_ij)/n_i).")
w("")
w("  Absorbing chain formulation:")
w("    Built Area forced absorbing (P_BB=1, P_Bj=0 for j≠B).")
w("    Q  = 6×6 transient sub-matrix.")
w("    N  = (I−Q)⁻¹  fundamental matrix.")
w("    E[T_i] = Σ_j N_{ij}  expected absorption time from class i.")
w("    SD[T_i] = sqrt( (2N−I)·E[T] − E[T]² )  intrinsic spread")
w("    of the first-passage-time distribution (NOT estimation")
w("    uncertainty — E[T] is exact given P).")
w("")
w("  Stationary π: left eigenvector of empirical P (not forced).")
w("  G-test + Frobenius norm for non-stationarity.")

# ── POOLED P ──────────────────────────────────────────────
w("")
w(SEP2)
w("  POOLED TRANSITION PROBABILITY MATRIX")
w(SEP2)
w("")
hdr = "  " + f"{'':>14}" + "".join(f"{lb:>14}" for lb in LABELS)
w(hdr)
for i in range(N_CLASSES):
    row = f"  {LABELS[i]:>14}"
    for j in range(N_CLASSES):
        row += f"{P_pooled[i,j]:>14.6f}"
    w(row)

w("")
w("  Asymptotic SE:")
w(hdr)
for i in range(N_CLASSES):
    row = f"  {LABELS[i]:>14}"
    for j in range(N_CLASSES):
        row += f"{SE_pooled[i,j]:>14.6f}"
    w(row)

w("")
w("  Row totals (n_i):")
rt = C_pooled.sum(axis=1)
for i in range(N_CLASSES):
    w(f"    {LABELS[i]:>14} : {int(rt[i]):>14,}")
w(f"  Total transitions: {int(C_pooled.sum()):,}")

# ── POOLED COUNTS ─────────────────────────────────────────
w("")
w(SEP2)
w("  POOLED COUNT MATRIX")
w(SEP2)
w("")
w(hdr)
for i in range(N_CLASSES):
    row = f"  {LABELS[i]:>14}"
    for j in range(N_CLASSES):
        row += f"{C_pooled[i,j]:>14,}"
    w(row)

# ── EIGENVALUES ───────────────────────────────────────────
w("")
w(SEP2)
w("  SPECTRAL DECOMPOSITION (pooled P)")
w(SEP2)
w("")
for k, ev in enumerate(ev_pooled):
    im_s = ""
    if np.abs(np.imag(ev)) > 1e-10:
        im_s = f"  {'+'if np.imag(ev)>=0 else ''}{np.imag(ev):.6f}i"
    w(f"    λ_{k+1} = {np.real(ev):>12.6f}{im_s}   (|λ| = {abs(ev):.6f})")
w(f"  Spectral gap : {gap_pooled:.6f}")
w(f"  Mixing time  : {mt_pooled:.2f} steps")

# ── STATIONARY π ──────────────────────────────────────────
w("")
w(SEP2)
w("  STATIONARY DISTRIBUTION π")
w(SEP2)
w("")
for i in range(N_CLASSES):
    bar = "█" * int(pi_pooled[i] * 50)
    w(f"    {LABELS[i]:>14} : {pi_pooled[i]:.6f}  {bar}")
w("")
w(f"  Long-run: {pi_pooled[BUILT_IDX]*100:.1f}% Built, "
  f"{pi_pooled[CODE_TO_IDX[2]]*100:.2f}% Trees.")

# ── ABSORBING CHAIN ───────────────────────────────────────
w("")
w(SEP2)
w("  ABSORBING CHAIN ANALYSIS  (Built Area = absorbing)")
w(SEP2)
w("")
w("  Transient sub-matrix Q:")
hdr_t = "  " + f"{'':>14}" + "".join(f"{TRANSIENT_LABELS[j]:>14}"
                                      for j in range(N_TRANSIENT))
w(hdr_t)
for i in range(N_TRANSIENT):
    row = f"  {TRANSIENT_LABELS[i]:>14}"
    for j in range(N_TRANSIENT):
        row += f"{Q_pooled[i,j]:>14.6f}"
    w(row)

w("")
w("  1-step absorption probability (1 − row sum of Q):")
for i in range(N_TRANSIENT):
    rs = Q_pooled[i].sum()
    w(f"    {TRANSIENT_LABELS[i]:>14} : {1-rs:.6f}")

w("")
w("  Fundamental matrix N = (I−Q)⁻¹:")
w(hdr_t)
for i in range(N_TRANSIENT):
    row = f"  {TRANSIENT_LABELS[i]:>14}"
    for j in range(N_TRANSIENT):
        row += f"{abs_N[i,j]:>14.4f}"
    w(row)

w("")
w("  EXPECTED ABSORPTION TIME  E[T_i] = Σ_j N_{ij}")
w("")
w(f"  {'Class':>14}  {'E[T] (yr)':>10}  {'SD[T] (yr)':>11}  Interpretation")
w(f"  {'-'*14}  {'-'*10}  {'-'*11}  {'-'*30}")
rank = np.argsort(abs_t)
for i in rank:
    note = ""
    if abs_t[i] == abs_t.min():
        note = "← fastest to commodification"
    elif abs_t[i] == abs_t.max():
        note = "← slowest (Water buffer)"
    w(f"  {TRANSIENT_LABELS[i]:>14}  {abs_t[i]:>10.2f}  {abs_sd[i]:>11.2f}  {note}")
w("")
w("  Note: E[T] is exact given P — no sampling uncertainty.")
w("  SD[T] is the intrinsic spread of individual pixel")
w("  trajectories (first-passage-time distribution), not")
w("  an error bar on the mean.  The distribution is highly")
w("  right-skewed, so SD > E[T] is expected for quasi-absorbing")
w("  chains with high self-retention in transient states.")

# ── ABSORPTION TIME PER PERIOD ────────────────────────────
w("")
w(SEP2)
w("  ABSORPTION TIME BY PERIOD  (E[T] in years)")
w(SEP2)
w("")
hdr_p = f"  {'Class':>14}" + "".join(f"{p:>12}" for p in periods)
w(hdr_p)
w("  " + "-"*14 + "-"*12*len(periods))
for i in range(N_TRANSIENT):
    row = f"  {TRANSIENT_LABELS[i]:>14}"
    for t in range(len(periods)):
        val = abs_t_per_period[t][i]
        if np.isnan(val) or val > 9999:
            row += f"{'—':>12}"
        else:
            row += f"{val:>12.1f}"
    w(row)
w("")
w("  Declining E[T] over time = accelerating enclosure.")

# ── PERIOD-BY-PERIOD P ────────────────────────────────────
for t in range(len(periods)):
    w("")
    w(SEP2)
    w(f"  PERIOD : {periods[t]}")
    w(SEP2)
    w("")
    w("  Count matrix:")
    w(hdr)
    for i in range(N_CLASSES):
        row = f"  {LABELS[i]:>14}"
        for j in range(N_CLASSES):
            row += f"{count_mats[t][i,j]:>14,}"
        w(row)
    w(f"  Total: {int(count_mats[t].sum()):,}")
    w("")
    w("  Probability matrix:")
    w(hdr)
    for i in range(N_CLASSES):
        row = f"  {LABELS[i]:>14}"
        for j in range(N_CLASSES):
            row += f"{prob_mats[t][i,j]:>14.6f}"
        w(row)
    w("")
    w("  SE matrix:")
    w(hdr)
    for i in range(N_CLASSES):
        row = f"  {LABELS[i]:>14}"
        for j in range(N_CLASSES):
            row += f"{se_mats[t][i,j]:>14.6f}"
        w(row)
    w("")
    ev = eigval_list[t]
    for k in range(len(ev)):
        im_s = ""
        if np.abs(np.imag(ev[k])) > 1e-10:
            im_s = f"  {'+'if np.imag(ev[k])>=0 else ''}{np.imag(ev[k]):.6f}i"
        w(f"    λ_{k+1} = {np.real(ev[k]):>12.6f}{im_s}"
          f"   (|λ| = {abs(ev[k]):.6f})")
    w(f"  Spectral gap : {gaps[t]:.6f}   Mixing time : {mix_times[t]:.2f}")

# ── STATIONARITY ──────────────────────────────────────────
w("")
w(SEP2)
w("  G-TEST FOR STATIONARITY")
w(SEP2)
w(f"  G = {G_stat:,.2f}   df = {G_df}   p = {G_pval:.2e}")
w(f"  Decision: "
  f"{'REJECT H₀ — non-stationary' if G_pval < 0.05 else 'FAIL TO REJECT H₀'}")
w("")
w("  Frobenius effect sizes ‖P_t − P_pooled‖_F :")
for t in range(len(periods)):
    w(f"    {periods[t]} : {frob_dists[t]:.6f}")
w(f"  Range: [{min(frob_dists):.6f}, {max(frob_dists):.6f}]  "
  f"Mean: {np.mean(frob_dists):.6f}")

# ── SELF-RETENTION ────────────────────────────────────────
w("")
w(SEP2)
w("  BUILT AREA SELF-RETENTION")
w(SEP2)
w("")
w(f"  {'Period':<12} {'P(B→B)':>10} {'±SE':>10} {'Leakage%':>10} {'n_built':>12}")
w(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
for t in range(len(periods)):
    pbb = self_ret[t]; sebb = self_ret_se[t]
    nb  = int(count_mats[t][BUILT_IDX].sum())
    w(f"  {periods[t]:<12} {pbb:>10.6f} {sebb:>10.6f} "
      f"{(1-pbb)*100:>9.2f}% {nb:>12,}")
w("")
w("  ~4% leakage = construction churn (Built ↔ Bare Ground),")
w("  not de-commodification.  Quasi-absorbing: once built,")
w("  land stays in the accumulation circuit.")

# ── STRUCTURAL NOTES ──────────────────────────────────────
w("")
w(SEP2)
w("  STRUCTURAL ZEROS")
w(SEP2)
w("")
zeros = [(LABELS[i], LABELS[j])
         for i in range(N_CLASSES) for j in range(N_CLASSES)
         if C_pooled[i, j] == 0]
for s, d in zeros:
    w(f"    {s:>14} → {d}")
w(f"  ({len(zeros)} zero entries)")

w("")
w(SEP2)
w("  SMALL-CLASS CAVEAT")
w(SEP2)
w("")
for i in range(N_CLASSES):
    n = int(rt[i])
    if n < 100_000:
        w(f"    {LABELS[i]:>14} : n = {n:>10,}  (interpret with caution)")

w("")
w(SEP)
w("  END OF REPORT")
w(SEP)

rpath = os.path.join(REPORTDIR, "markov_report.txt")
with open(rpath, "w") as f:
    f.write("\n".join(L))
print(f"  → {rpath}")
print("\nDone.")
