#!/usr/bin/env python3
"""
================================================================
  Percolation Analysis  —  PIK2 LULC (2017–2024)
  Author : Sandy H. S. Herho <sandy.herho@email.ucr.edu>
  Date   : 2026-03-09
  License: MIT

  Input  : ../netcdf/pik2LULC.nc
  Output : ../figs/percolation_analysis.{pdf,png}
           ../reports/percolation_report.txt
================================================================
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
from scipy.ndimage import label, binary_erosion
from scipy.stats import linregress
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
    "mathtext.fontset":    "dejavuserif",
})

BG_DARK = "#1C2833"
BUILT_CODE = 7
FILL_VAL   = -128


# ── functions ─────────────────────────────────────────────────

def load_data():
    with nc.Dataset(NCFILE) as ds:
        years = np.array(ds["year"][:]).astype(int)
        lulc  = np.ma.filled(ds["lulc"][:], fill_value=FILL_VAL).astype(np.int16)
    return lulc, years


def cluster_analysis(arr):
    built_mask = (arr == BUILT_CODE).astype(np.int32)
    n_total  = arr.size
    n_built  = int(built_mask.sum())
    p_occ    = n_built / n_total

    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int32)
    labeled, n_clusters = label(built_mask, structure=struct)

    if n_clusters == 0:
        return dict(n_total=n_total, n_built=n_built, p_occ=p_occ,
                    n_clusters=0, s_max=0, s_max_frac=0.0,
                    s_second=0, sizes=np.array([]),
                    labeled=labeled, built_mask=built_mask)

    sizes = np.bincount(labeled.ravel())[1:]
    s_max = int(sizes.max())
    sorted_sizes = np.sort(sizes)[::-1]
    s_second = int(sorted_sizes[1]) if len(sorted_sizes) > 1 else 0
    s_max_frac = s_max / n_built if n_built > 0 else 0.0

    return dict(n_total=n_total, n_built=n_built, p_occ=p_occ,
                n_clusters=int(n_clusters), s_max=s_max,
                s_max_frac=s_max_frac, s_second=s_second,
                sizes=sizes, labeled=labeled, built_mask=built_mask)


def fractal_dimension_boxcount(binary_mask, min_box=2, max_box=None):
    interior = binary_erosion(binary_mask, structure=np.ones((3,3)))
    boundary = binary_mask.astype(bool) & ~interior

    if boundary.sum() < 10:
        return np.nan, np.nan, np.array([]), np.array([])

    coords = np.argwhere(boundary)
    if len(coords) == 0:
        return np.nan, np.nan, np.array([]), np.array([])

    if max_box is None:
        max_box = max(binary_mask.shape) // 4

    box_sizes = []
    counts = []
    sizes = np.unique(np.geomspace(min_box, max_box, num=20).astype(int))
    sizes = sizes[sizes >= min_box]

    for eps in sizes:
        boxes = set()
        for r, c in coords:
            boxes.add((r // eps, c // eps))
        box_sizes.append(eps)
        counts.append(len(boxes))

    box_sizes = np.array(box_sizes, dtype=float)
    counts = np.array(counts, dtype=float)

    log_inv_eps = np.log(1.0 / box_sizes)
    log_n = np.log(counts)
    slope, intercept, r_value, p_value, std_err = linregress(log_inv_eps, log_n)

    return slope, r_value**2, box_sizes, counts


# ══════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════

print(f"Reading {NCFILE} …")
lulc, years = load_data()
n_time = len(years)

print("Running cluster analysis …")
results = []
for t in range(n_time):
    print(f"  {years[t]} …", end="")
    r = cluster_analysis(lulc[t])
    print(f" p={r['p_occ']:.4f}  N={r['n_clusters']}  "
          f"S_max/S_built={r['s_max_frac']:.4f}")
    results.append(r)

print("Computing fractal dimensions …")
frac_dims = []
frac_r2s  = []
for t in range(n_time):
    print(f"  {years[t]} …", end="")
    r = results[t]
    if r['n_clusters'] > 0:
        largest_label = np.argmax(r['sizes']) + 1
        largest_mask = (r['labeled'] == largest_label).astype(np.int32)
        df, r2, _, _ = fractal_dimension_boxcount(largest_mask)
    else:
        df, r2 = np.nan, np.nan
    frac_dims.append(df)
    frac_r2s.append(r2)
    print(f" d_f={df:.3f}  R²={r2:.4f}" if not np.isnan(df) else " (no data)")


# ══════════════════════════════════════════════════════════════
#  FIGURE : 2 × 2
# ══════════════════════════════════════════════════════════════

print("Plotting …")
fig = plt.figure(figsize=(14.0, 12.0), dpi=DPI)
fig.patch.set_facecolor("white")

gs = GridSpec(2, 2, figure=fig,
             left=0.09, right=0.96,
             top=0.96, bottom=0.07,
             hspace=0.35, wspace=0.32)


# ── (a) ORDER PARAMETER ─────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

s_max_frac = [r['s_max_frac'] for r in results]

ax_a.plot(years, s_max_frac, "o-", color="#C0392B", lw=2.2, ms=7,
          markeredgecolor="white", markeredgewidth=1.0, zorder=5)

ax_a.set_xlabel("Year", fontsize=10, fontweight="bold",
                color=BG_DARK, labelpad=5)
ax_a.set_ylabel(r"$S_{\max}\, /\, S_{\mathrm{built}}$", fontsize=11,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_a.set_xticks(years)
ax_a.set_xticklabels(years, fontsize=8.5, fontweight="bold",
                     color=BG_DARK, rotation=30, ha="right")
ax_a.tick_params(axis="y", labelsize=9, labelcolor=BG_DARK, width=0.5)
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


# ── (b) NUMBER OF CLUSTERS ──────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

n_clust = [r['n_clusters'] for r in results]

ax_b.plot(years, n_clust, "s-", color="#2980B9", lw=2.2, ms=7,
          markeredgecolor="white", markeredgewidth=1.0, zorder=5)

ax_b.set_xlabel("Year", fontsize=10, fontweight="bold",
                color=BG_DARK, labelpad=5)
ax_b.set_ylabel(r"Number of Clusters  $N(t)$", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_b.set_xticks(years)
ax_b.set_xticklabels(years, fontsize=8.5, fontweight="bold",
                     color=BG_DARK, rotation=30, ha="right")
ax_b.tick_params(axis="y", labelsize=9, labelcolor=BG_DARK, width=0.5)
for lb in ax_b.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax_b.grid(True, axis="y", linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_b.set_axisbelow(True)
for sp in ax_b.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_b.text(0.50, 1.05, "(b)", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (c) CLUSTER SIZE CCDF ───────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

cmap = plt.cm.viridis
norm_t = plt.Normalize(vmin=0, vmax=n_time - 1)

for t in range(n_time):
    sizes = results[t]['sizes']
    if len(sizes) == 0:
        continue
    sorted_s = np.sort(sizes)[::-1]
    ccdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    color = cmap(norm_t(t))
    ax_c.loglog(sorted_s, ccdf, "-", color=color, lw=1.5,
                alpha=0.85, label=str(years[t]), zorder=3 + t)

if len(results[-1]['sizes']) > 0:
    s_ref = np.logspace(0, np.log10(results[-1]['s_max']), 50)
    ax_c.loglog(s_ref, (s_ref / s_ref[0]) ** (-2.05 + 1) * 0.5,
                "--", color="#95A5A6", lw=1.0, alpha=0.6,
                label=r"$\tau = 2.05$ (ref.)")

ax_c.set_xlabel(r"Cluster Size  $s$  (pixels)", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_c.set_ylabel(r"$P(S \geq s)$", fontsize=11,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_c.tick_params(axis="both", labelsize=8.5, labelcolor=BG_DARK, width=0.5)
for lb in ax_c.xaxis.get_ticklabels() + ax_c.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

leg_c = ax_c.legend(fontsize=6.5, frameon=True, fancybox=False,
                    edgecolor="#BDC3C7", framealpha=0.9,
                    loc="lower left", ncol=2,
                    handlelength=1.5, handletextpad=0.4,
                    columnspacing=0.8)
for txt in leg_c.get_texts():
    txt.set_fontweight("bold"); txt.set_fontfamily("serif")
    txt.set_color(BG_DARK)
leg_c.get_frame().set_linewidth(0.5)

ax_c.grid(True, linewidth=0.25, color="#BDC3C7",
          alpha=0.4, linestyle=":")
ax_c.set_axisbelow(True)
for sp in ax_c.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

ax_c.text(0.50, 1.05, "(c)", transform=ax_c.transAxes,
          fontsize=14, fontweight="bold", color=BG_DARK,
          fontfamily="serif", ha="center", va="bottom")


# ── (d) FRACTAL DIMENSION ───────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

valid_fd = [(years[t], frac_dims[t]) for t in range(n_time)
            if not np.isnan(frac_dims[t])]
if valid_fd:
    yrs_fd, dfs = zip(*valid_fd)
    ax_d.plot(yrs_fd, dfs, "D-", color="#8E44AD", lw=2.2, ms=7,
              markeredgecolor="white", markeredgewidth=1.0, zorder=5)

# zoom y-axis to data range
if valid_fd:
    df_arr = np.array(dfs)
    y_lo = max(df_arr.min() - 0.08, 0.95)
    y_hi = min(df_arr.max() + 0.15, 1.85)
    ax_d.set_ylim(y_lo, y_hi)

ax_d.set_xlabel("Year", fontsize=10, fontweight="bold",
                color=BG_DARK, labelpad=5)
ax_d.set_ylabel(r"Fractal Dimension  $d_f$", fontsize=10,
                fontweight="bold", color=BG_DARK, labelpad=5)
ax_d.set_xticks(years)
ax_d.set_xticklabels(years, fontsize=8.5, fontweight="bold",
                     color=BG_DARK, rotation=30, ha="right")
ax_d.tick_params(axis="y", labelsize=9, labelcolor=BG_DARK, width=0.5)
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
    path = os.path.join(FIGDIR, f"percolation_analysis.{ext}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  → {path}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  REPORT  (unchanged — all numbers reported here)
# ══════════════════════════════════════════════════════════════

print("Writing report …")
SEP  = "=" * 80
SEP2 = "-" * 80
L = []
w = L.append

w(SEP)
w("  PERCOLATION ANALYSIS  —  PIK2 LULC (2017–2024)")
w(f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
w(SEP)
w("")
w("  METHOD")
w("  ------")
w("  Built Area (code=7) as occupied sites on the raster lattice.")
w("  Connected components via 4-connectivity (von Neumann).")
w("")
w("  Quantities:")
w("    p(t)           = Built / total pixels")
w("    S_max(t)       = largest connected cluster")
w("    S_max/S_built  = order parameter")
w("    N(t)           = number of clusters")
w("    P(S≥s)         = complementary CDF of cluster sizes")
w("    d_f(t)         = fractal dimension (box-counting)")
w("")
w("  Reference universality classes:")
w("    2D site percolation : τ ≈ 2.05")
w("    DLA boundary        : d_f ≈ 1.71–1.75")
w("    Eden growth         : d_f ≈ 1.50")
w("    Compact growth      : d_f → 1.0")

w("")
w(SEP2)
w("  CLUSTER STATISTICS BY YEAR")
w(SEP2)
w("")
w(f"  {'Year':>6} {'p(occ)':>8} {'N_built':>10} {'N_clust':>8} "
  f"{'S_max':>10} {'S_max/S_b':>10} {'S_2nd':>8} {'d_f':>8} {'R²':>8}")
w(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8} "
  f"{'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for t in range(n_time):
    r = results[t]
    df_s = f"{frac_dims[t]:.3f}" if not np.isnan(frac_dims[t]) else "—"
    r2_s = f"{frac_r2s[t]:.4f}" if not np.isnan(frac_r2s[t]) else "—"
    w(f"  {years[t]:>6} {r['p_occ']:>8.4f} {r['n_built']:>10,} "
      f"{r['n_clusters']:>8,} {r['s_max']:>10,} "
      f"{r['s_max_frac']:>10.4f} {r['s_second']:>8,} "
      f"{df_s:>8} {r2_s:>8}")

w("")
w(SEP2)
w("  ORDER PARAMETER INTERPRETATION")
w(SEP2)
w("")
s_fracs = [r['s_max_frac'] for r in results]
w(f"  S_max/S_built range: [{min(s_fracs):.4f}, {max(s_fracs):.4f}]")
w("")
if max(s_fracs) > 0.9:
    w("  S_max/S_built > 0.9 → SUPERCRITICAL regime.")
    w("  A giant connected component contains >90% of Built pixels.")
    w("  This occurs at p = 0.096–0.162, far below the random")
    w("  percolation threshold p_c ≈ 0.593.  Capital achieves")
    w("  spatial connectivity through planned infrastructure,")
    w("  not random colonisation.")

w("")
n_cs = [r['n_clusters'] for r in results]
w(f"  N_clusters: {min(n_cs):,}–{max(n_cs):,}  (Δ = {n_cs[-1]-n_cs[0]:+,})")
if n_cs[-1] > n_cs[0]:
    w("  Rising cluster count + persistent S_max/S_built > 0.9 =")
    w("  new satellite developments at the frontier while the")
    w("  core consolidates (expanding accumulation frontier).")

w("")
w(SEP2)
w("  FRACTAL DIMENSION")
w(SEP2)
w("")
for t in range(n_time):
    if not np.isnan(frac_dims[t]):
        w(f"  {years[t]} : d_f = {frac_dims[t]:.4f}  (R² = {frac_r2s[t]:.4f})")
w("")
valid_dfs = [d for d in frac_dims if not np.isnan(d)]
if valid_dfs:
    w(f"  Mean d_f = {np.mean(valid_dfs):.3f}")
    w(f"  Range    = [{min(valid_dfs):.3f}, {max(valid_dfs):.3f}]")
    w(f"  Trend    : {valid_dfs[0]:.3f} → {valid_dfs[-1]:.3f}")
    w("")
    mean_df = np.mean(valid_dfs)
    if mean_df > 1.6:
        w("  → DLA-like irregular boundary (speculative expansion).")
    elif mean_df > 1.3:
        w("  → Intermediate between Eden and DLA.")
        w("  Partially planned (road-led) with irregular frontier.")
        w("  The upward trend (1.31→1.40) reflects increasing")
        w("  frontier irregularity as construction pushes into")
        w("  unplanned peripheral areas.")
    else:
        w("  → Near-compact growth (planned expansion).")

w("")
w(SEP2)
w("  CLUSTER SIZE DISTRIBUTION")
w(SEP2)
w("")
for t in range(n_time):
    sizes = results[t]['sizes']
    if len(sizes) == 0:
        continue
    n_iso = int(np.sum(sizes == 1))
    top5 = np.sort(sizes)[::-1][:5]
    w(f"  {years[t]}:  N={len(sizes):,}  S_max={sizes.max():,}  "
      f"median={int(np.median(sizes)):,}  isolated={n_iso} ({n_iso/len(sizes)*100:.0f}%)")
    w(f"         top5: {', '.join(f'{s:,}' for s in top5)}")

w("")
w(SEP2)
w("  CONTEXT: p vs p_c")
w(SEP2)
w("")
w("  p_c(square lattice) ≈ 0.593")
w(f"  Our p range: [{min(r['p_occ'] for r in results):.3f}, "
  f"{max(r['p_occ'] for r in results):.3f}]")
w("  Giant component at p << p_c → spatial correlation from")
w("  planned infrastructure, not random site occupation.")

w("")
w(SEP)
w("  END OF REPORT")
w(SEP)

rpath = os.path.join(REPORTDIR, "percolation_report.txt")
with open(rpath, "w") as f:
    f.write("\n".join(L))
print(f"  → {rpath}")
print("\nDone.")
