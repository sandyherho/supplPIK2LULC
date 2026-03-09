#!/usr/bin/env python3
"""
================================================================
  Ternary Simplex Trajectory  —  PIK2 LULC (2017–2024)
  Author : Sandy H. S. Herho <sandy.herho@email.ucr.edu>
  Date   : 2026-03-09
  License: MIT

  Input  : ../netcdf/pik2LULC.nc
  Output : ../figs/simplex_trajectory.{pdf,png}
           ../reports/simplex_report.txt

  LAND-ONLY analysis (Water/ocean excluded):
    Commons  = Trees + Flooded Veg. + Rangeland
    Agrarian = Crops
    Capital  = Built Area + Bare Ground
================================================================
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ── LULC codes ────────────────────────────────────────────────
FILL_VAL = -128

COMMONS_CODES  = [2, 4, 11]
AGRARIAN_CODES = [5]
CAPITAL_CODES  = [7, 8]
ALL_LAND_CODES = COMMONS_CODES + AGRARIAN_CODES + CAPITAL_CODES

CAT_NAMES  = ["Commons", "Agrarian", "Capital"]
CAT_COLORS = ["#2E86C1", "#F39C12", "#C0392B"]

FINE_LABELS = {2: "Trees", 4: "Flooded Veg.", 11: "Rangeland",
               5: "Crops", 7: "Built Area", 8: "Bare Ground"}


# ── functions ─────────────────────────────────────────────────

def load_data():
    with nc.Dataset(NCFILE) as ds:
        years = np.array(ds["year"][:]).astype(int)
        lulc  = np.ma.filled(ds["lulc"][:], fill_value=FILL_VAL).astype(np.int16)
    return lulc, years


def aggregate_land_only(arr):
    flat = arr.ravel()
    counts = np.zeros(3, dtype=np.int64)
    for code in COMMONS_CODES:
        counts[0] += np.sum(flat == code)
    for code in AGRARIAN_CODES:
        counts[1] += np.sum(flat == code)
    for code in CAPITAL_CODES:
        counts[2] += np.sum(flat == code)
    total = counts.sum()
    if total == 0:
        return np.array([1./3, 1./3, 1./3]), 0
    return counts.astype(np.float64) / total, int(total)


def fine_grained_land(arr):
    flat = arr.ravel()
    total = sum(int(np.sum(flat == c)) for c in ALL_LAND_CODES)
    if total == 0:
        return {c: 0.0 for c in ALL_LAND_CODES}
    return {c: int(np.sum(flat == c)) / total for c in ALL_LAND_CODES}


def fisher_rao_3(p, q):
    bc = np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0)))
    return 2.0 * np.arccos(np.clip(bc, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════

print(f"Reading {NCFILE} …")
lulc, years = load_data()
n_time = len(years)

print("Computing land-only distributions …")
dists = []
n_land_pixels = []
fine_dists = []
for t in range(n_time):
    d, nl = aggregate_land_only(lulc[t])
    dists.append(d)
    n_land_pixels.append(nl)
    fine_dists.append(fine_grained_land(lulc[t]))
    print(f"  {years[t]}: Commons={d[0]*100:.1f}%  Agrarian={d[1]*100:.1f}%  "
          f"Capital={d[2]*100:.1f}%  (n_land={nl:,})")

dists = np.array(dists)

fr_dists = np.array([fisher_rao_3(dists[t], dists[t+1])
                      for t in range(n_time - 1)])
arc_cum = np.insert(np.cumsum(fr_dists), 0, 0.0)
total_displacement = fisher_rao_3(dists[0], dists[-1])


# ══════════════════════════════════════════════════════════════
#  FIGURE : single panel
# ══════════════════════════════════════════════════════════════

print("Plotting …")
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)
fig.patch.set_facecolor("white")

for i in range(3):
    ax.plot(years, dists[:, i] * 100, "o-",
            color=CAT_COLORS[i], lw=2.4, ms=8,
            markeredgecolor="white", markeredgewidth=1.2,
            label=CAT_NAMES[i], zorder=5 - i)

ax.set_xlabel("Year", fontsize=11, fontweight="bold",
              color=BG_DARK, labelpad=6)
ax.set_ylabel("Fraction of Land Area  (%)", fontsize=11,
              fontweight="bold", color=BG_DARK, labelpad=6)
ax.set_xticks(years)
ax.set_xticklabels(years, fontsize=9.5, fontweight="bold",
                   color=BG_DARK, rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=9.5, labelcolor=BG_DARK, width=0.5)
for lb in ax.yaxis.get_ticklabels():
    lb.set_fontweight("bold"); lb.set_fontfamily("serif")

ax.grid(True, axis="y", linewidth=0.25, color="#BDC3C7",
        alpha=0.4, linestyle=":")
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_edgecolor(BG_DARK); sp.set_linewidth(0.7)

# legend outside at the bottom
leg = ax.legend(fontsize=10, frameon=True, fancybox=False,
                edgecolor="#BDC3C7", framealpha=0.95,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                handlelength=2.0, handletextpad=0.6,
                columnspacing=2.5)
for txt in leg.get_texts():
    txt.set_fontweight("bold"); txt.set_fontfamily("serif")
    txt.set_color(BG_DARK)
leg.get_frame().set_linewidth(0.5)

for ext in ("pdf", "png"):
    path = os.path.join(FIGDIR, f"simplex_trajectory.{ext}")
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
w("  TERNARY SIMPLEX TRAJECTORY  —  PIK2 LULC (2017–2024)")
w(f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
w(SEP)
w("")
w("  LAND-ONLY MARXIAN AGGREGATION")
w("  -----------------------------")
w("  Water (ocean, code=1) is EXCLUDED — it constitutes ~70%")
w("  of the bounding box but is not subject to enclosure.")
w("  All fractions computed relative to land pixels only.")
w("")
w("  Commons  = Trees + Flooded Veg. + Rangeland")
w("  Agrarian = Crops")
w("  Capital  = Built Area + Bare Ground")
w("")
w("  Bare Ground ∈ Capital: Markov analysis shows")
w("  P(Bare Ground → Built) = 0.194 — it is construction-")
w("  phase transitional land, not natural bare soil.")

# ── DISTRIBUTIONS ─────────────────────────────────────────
w("")
w(SEP2)
w("  AGGREGATED LAND-ONLY DISTRIBUTIONS")
w(SEP2)
w("")
w(f"  {'Year':>6} {'Commons%':>10} {'Agrarian%':>10} {'Capital%':>10} {'n_land':>12}")
w(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
for t in range(n_time):
    w(f"  {years[t]:>6} {dists[t,0]*100:>10.2f} {dists[t,1]*100:>10.2f} "
      f"{dists[t,2]*100:>10.2f} {n_land_pixels[t]:>12,}")

# ── FINE-GRAINED ──────────────────────────────────────────
w("")
w(SEP2)
w("  FINE-GRAINED LAND FRACTIONS  (% of land)")
w(SEP2)
w("")
fine_labs = [FINE_LABELS[c] for c in ALL_LAND_CODES]
hdr_f = f"  {'Year':>6}" + "".join(f"{lb:>14}" for lb in fine_labs)
w(hdr_f)
w("  " + "-"*6 + "-"*14*len(fine_labs))
for t in range(n_time):
    row = f"  {years[t]:>6}"
    for c in ALL_LAND_CODES:
        row += f"{fine_dists[t][c]*100:>14.3f}"
    w(row)

# ── SECULAR TRENDS ────────────────────────────────────────
w("")
w(SEP2)
w("  SECULAR TRENDS  (2017 → 2024)")
w(SEP2)
w("")
for i, name in enumerate(CAT_NAMES):
    d0, d1 = dists[0, i], dists[-1, i]
    delta = d1 - d0
    pct_rel = delta / d0 * 100 if d0 > 0 else 0
    w(f"  {name:>10} : {d0*100:.1f}% → {d1*100:.1f}%  "
      f"(Δ = {delta*100:+.1f} pp,  {pct_rel:+.0f}% relative)")
w("")
w(f"  Capital grew from {dists[0,2]*100:.1f}% to {dists[-1,2]*100:.1f}%"
  f" of land (+{(dists[-1,2]/dists[0,2]-1)*100:.0f}% relative).")
w(f"  Agrarian declined from {dists[0,1]*100:.1f}% to {dists[-1,1]*100:.1f}%"
  f" (−{abs(dists[-1,1]/dists[0,1]-1)*100:.0f}% relative).")
w(f"  Commons declined from {dists[0,0]*100:.1f}% to {dists[-1,0]*100:.1f}%"
  f" (−{abs(dists[-1,0]/dists[0,0]-1)*100:.0f}% relative).")
w("")
w("  INTERPRETATION:")
w("  The dominant secular transfer is Agrarian → Capital.")
w("  Cropland lost 13.2 pp while Capital gained 14.1 pp —")
w("  nearly 1:1 replacement.  Commons loss is small (0.9 pp)")
w("  because natural vegetation was already a minor fraction.")
w("  This sequence — peasant dispossession preceding ecological")
w("  destruction — is characteristic of Marxian primitive")
w("  accumulation in coastal mega-development contexts.")
w("")
w("  The Capital–Agrarian crossover occurs ~2020–2021:")
w("  the moment commodified land exceeds productive agricultural")
w("  land.  This structural inversion marks the transition from")
w("  a predominantly agrarian to a predominantly capitalised")
w("  landscape — the completion of subsumption.")

# ── FISHER-RAO ────────────────────────────────────────────
w("")
w(SEP2)
w("  FISHER-RAO GEODESIC DISTANCES  (3-simplex, land-only)")
w(SEP2)
w("")
w(f"  {'Period':<12} {'d_FR (rad)':>12} {'Cum. arc':>12}")
w(f"  {'-'*12} {'-'*12} {'-'*12}")
for t in range(n_time - 1):
    w(f"  {years[t]}→{years[t+1]:<5} {fr_dists[t]:>12.6f} {arc_cum[t+1]:>12.6f}")
w("")
w(f"  Total arc length    : {arc_cum[-1]:.6f} rad")
w(f"  Direct displacement : {total_displacement:.6f} rad")
sinuosity = arc_cum[-1] / total_displacement if total_displacement > 0 else np.inf
w(f"  Sinuosity (arc/disp): {sinuosity:.4f}")
w("")
w(f"  Sinuosity = {sinuosity:.2f} → {'highly' if sinuosity > 2 else 'moderately'}"
  f" non-linear path.")
w("  The landscape zigzags through Agrarian expansion/contraction")
w("  (seasonal cycles, construction pauses) while drifting")
w("  secularly toward Capital.")

# ── VELOCITY ──────────────────────────────────────────────
w("")
w(SEP2)
w("  VELOCITY AND ACCELERATION")
w(SEP2)
w("")
accel = np.diff(fr_dists)
w(f"  {'Period':<12} {'Velocity':>12} {'Accel.':>12}")
w(f"  {'-'*12} {'-'*12} {'-'*12}")
for t in range(n_time - 1):
    a_str = f"{accel[t-1]:>+12.6f}" if t > 0 else f"{'—':>12}"
    w(f"  {years[t]}→{years[t+1]:<5} {fr_dists[t]:>12.6f} {a_str}")
w("")
w(f"  Peak velocity: {years[np.argmax(fr_dists)]}→{years[np.argmax(fr_dists)+1]}"
  f" ({fr_dists.max():.6f} rad/yr)")
w(f"  Mean velocity: {np.mean(fr_dists):.6f} rad/yr")

# ── DIRECTION ─────────────────────────────────────────────
w("")
w(SEP2)
w("  DIRECTION OF MOVEMENT  (Δ in percentage points)")
w(SEP2)
w("")
w(f"  {'Period':<12} {'ΔCommons':>10} {'ΔAgrarian':>10} {'ΔCapital':>10}  Dominant")
w(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}  {'-'*16}")
cap_up = 0
for t in range(n_time - 1):
    dd = dists[t+1] - dists[t]
    dom = np.argmax(np.abs(dd))
    d_dir = "↑" if dd[dom] > 0 else "↓"
    if dd[2] > 0: cap_up += 1
    w(f"  {years[t]}→{years[t+1]:<5} {dd[0]*100:>+10.2f} {dd[1]*100:>+10.2f} "
      f"{dd[2]*100:>+10.2f}  {CAT_NAMES[dom]} {d_dir}")
w("")
w(f"  Capital ↑ in {cap_up}/{n_time-1} periods.")
w("  Only 2018→2019 shows Capital↓ (−3.2 pp), coinciding")
w("  with initial land preparation → temporary Agrarian reclassification.")

# ── CAPITAL DECOMPOSITION ─────────────────────────────────
w("")
w(SEP2)
w("  CAPITAL DECOMPOSITION  (% of land)")
w(SEP2)
w("")
w(f"  {'Year':>6} {'Built%':>10} {'BareGnd%':>10} {'Capital%':>10} {'Built/Cap':>10}")
w(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for t in range(n_time):
    fg = fine_dists[t]
    pb = fg[7] * 100
    pg = fg[8] * 100
    pc = pb + pg
    r  = pb / pc if pc > 0 else 0
    w(f"  {years[t]:>6} {pb:>10.2f} {pg:>10.2f} {pc:>10.2f} {r:>10.4f}")
w("")
w("  Built/Capital ratio:")
w("  2017: 0.98 → most Capital is completed development.")
w("  2023: 0.90 → 10% of Capital is active construction.")
w("  2024: 0.96 → construction catching up to clearance.")
w("  The dip in 2023 coincides with the most aggressive")
w("  land clearance phase (pre-PSN designation, March 2024).")

# ── COMMONS DECOMPOSITION ─────────────────────────────────
w("")
w(SEP2)
w("  COMMONS DECOMPOSITION  (% of land)")
w(SEP2)
w("")
w(f"  {'Year':>6} {'Trees%':>10} {'FloodVeg%':>10} {'Rangeland%':>10} {'Commons%':>10}")
w(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for t in range(n_time):
    fg = fine_dists[t]
    pt = fg[2] * 100; pf = fg[4] * 100; pr = fg[11] * 100
    w(f"  {years[t]:>6} {pt:>10.2f} {pf:>10.2f} {pr:>10.2f} {dists[t,0]*100:>10.2f}")
w("")
w("  Commons oscillates due to Flooded Veg. and Rangeland")
w("  variability (tidal/seasonal effects on coastal mangrove).")
w("  Trees remain persistently low (<1.2%) — near-total")
w("  deforestation was already accomplished before 2017.")

w("")
w(SEP)
w("  END OF REPORT")
w(SEP)

rpath = os.path.join(REPORTDIR, "simplex_report.txt")
with open(rpath, "w") as f:
    f.write("\n".join(L))
print(f"  → {rpath}")
print("\nDone.")
