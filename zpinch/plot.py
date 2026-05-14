#!/usr/bin/env python3
"""
Comparison Plots: Reaction Rates vs Fission Blanket Location
=============================================================

Reads the JSON results from both the UO₂ and U-10Zr Z-pinch parametric
sweeps and produces six separate figures:

    1. tbr.png/pdf              — Tritium breeding ratio
    2. u238ng.png/pdf           — U-238 (n,gamma) capture rate
    3. pu239.png/pdf            — Pu-239 production rate [kg/yr]
    4. am241.png/pdf            — Am-241 fission, (n,gamma), and (n,Xn) rates
    5. fuel_mass.png/pdf        — Spent fuel mass in blanket [kg]
    6. kinf.png/pdf             — k-inf of fuel zone
    7. heating.png/pdf          — Total, fission, and fusion thermal power [W]

Also exports Am-241 data to figures/am241_data.csv.

Both fuel types are overlaid on each figure for direct comparison.

Usage:
    python plot.py

Expects the output directories produced by uo2_zpinch.py and uzr_zpinch.py
to exist in the current working directory (or edit the paths below).
"""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

"""
MATPLOTLIB SETTINGS
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Fonts
# try:
#     font_path = './Python/fonts/DIN-Regular.ttf' # DIN-Regular.ttf' # './Python/arial.ttf'
#     fm.fontManager.addfont(font_path)
#     prop = fm.FontProperties(fname=font_path)
#     plt.rcParams['font.family'] = prop.get_name()
# except:
font_path = './figures/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['pdf.fonttype'] = 42

# Ticks
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.major.size']  = 3.5
plt.rcParams['ytick.major.size']  = 3.5
plt.rcParams['xtick.minor.size']  = 2.0
plt.rcParams['ytick.minor.size']  = 2.0
plt.rcParams['axes.linewidth']    = 0.5
plt.rcParams['grid.color'] = '#DBDBDB'
plt.rcParams['grid.linewidth'] = 0.5

# Font sizes
plt.rcParams['font.size']        =  8
plt.rcParams['axes.titlesize']   =  9
plt.rcParams['axes.labelsize']   =  8
plt.rcParams['xtick.labelsize']  =  7
plt.rcParams['ytick.labelsize']  =  7
plt.rcParams['legend.fontsize']  =  7
plt.rcParams['figure.titlesize'] = 10

# ==============================================================================
# Configuration — point these at the result directories from each run
# ==============================================================================
UO2_OUTPUT_DIR = "uo2_zpinch_results_shell20cm"
UZR_OUTPUT_DIR = "uzr_zpinch_results_shell20cm"

UO2_JSON = os.path.join(UO2_OUTPUT_DIR, "uo2_zpinch_all_results.json")
UZR_JSON = os.path.join(UZR_OUTPUT_DIR, "uzr_zpinch_all_results.json")

# Labels for the legend
UO2_LABEL = r"UO$_2$"
UZR_LABEL = r"U-10Zr"

XLABEL = "Fission blanket midpoint — radial depth [cm]"

# --- Pu-239 production rate conversion ---
# Fusion thermal power [W] from Zap Energy reactor concept (slide 14)
P_FUSION_W = 200e6                        # 200 MW thermal
E_DT_J     = 17.6e6 * 1.602176634e-19    # 17.6 MeV per D-T reaction [J]
S_NEUTRON  = P_FUSION_W / E_DT_J         # source neutrons per second
M_PU239    = 239.0522                     # g/mol
N_A        = 6.02214076e23               # Avogadro
SEC_PER_YR = 365.25 * 24 * 3600          # s/yr

# --- Blanket geometry for fuel mass calculation ---
R_INNER              = 1.0     # Inner blanket radius [cm]
FUEL_SHELL_THICKNESS = 20.0    # Radial thickness of fuel shell [cm]
BLANKET_HEIGHT       = 150.0   # Full blanket height [cm]
FUEL_VOL_FRAC        = 0.30    # Volume fraction of fuel in homogenized zone

# Spent fuel densities [g/cm³]
RHO_UO2  = 10.48   # UO₂ pellet at 95.5% TD (AP1000 PWR)
RHO_UZR  = 11.0    # U-10Zr metallic at ~75% smear density (4S-type)


# ==============================================================================
# Helper: read neutron-balance CSV and return summed U238 (n,gamma)
# ==============================================================================
def _read_nbal_csv(path):
    """Read a neutron-balance depth-profile CSV and return dict of arrays."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for key in rows[0]:
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except (ValueError, KeyError):
            pass
    return data


def _get_nbal_column_sum(result, col_name):
    """
    Sum a single depth-resolved column from the neutron-balance CSV.
    Returns 0.0 if the case has no fuel or the column is missing.
    """
    if result.get("no_fuel", False):
        return 0.0
    nbal_path = result.get("nbal_csv")
    if nbal_path is None or not os.path.isfile(nbal_path):
        return 0.0
    data = _read_nbal_csv(nbal_path)
    col = data.get(col_name)
    if col is None:
        return 0.0
    return float(col.sum())


def get_u238_ngamma(result):
    """Sum the depth-resolved U238 (n,gamma) rate."""
    return _get_nbal_column_sum(result, "U238_ngamma")


# ==============================================================================
# Load and extract
# ==============================================================================
def load_series(json_path):
    """
    Load a results JSON and return a dict of arrays,
    filtering out the pure-PbLi baseline case.
    """
    with open(json_path, "r") as f:
        all_results = json.load(f)

    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]

    midpoints   = np.array([r["shell_midpoint_cm"] for r in fuel_results])
    tbr         = np.array([r.get("TBR_mean", 0.0) for r in fuel_results])
    tbr_err     = np.array([r.get("TBR_std", 0.0)  for r in fuel_results])
    u238ng      = np.array([get_u238_ngamma(r)      for r in fuel_results])
    am241_fiss  = np.array([_get_nbal_column_sum(r, "Am241_fission")
                            for r in fuel_results])
    am241_ng    = np.array([_get_nbal_column_sum(r, "Am241_ngamma")
                            for r in fuel_results])
    am241_n2n   = np.array([_get_nbal_column_sum(r, "Am241_n2n")
                            for r in fuel_results])
    am241_n3n   = np.array([_get_nbal_column_sum(r, "Am241_n3n")
                            for r in fuel_results])
    am241_nXn   = am241_n2n + am241_n3n
    k_inf       = np.array([r.get("k_inf", 0.0)     for r in fuel_results])
    k_inf_std   = np.array([r.get("k_inf_std", 0.0)  for r in fuel_results])

    # --- Heating & fission-Q tallies (sum over all zones) ---
    def _sum_zones(d):
        """Sum all zone values in a heating/fisq dict."""
        if d is None:
            return 0.0
        return sum(d.values())

    heating_W   = np.array([_sum_zones(r.get("heating_W"))  for r in fuel_results])
    fisq_W      = np.array([_sum_zones(r.get("fisq_W"))     for r in fuel_results])
    fusion_W    = heating_W - fisq_W

    return {
        "midpoints": midpoints,
        "tbr": tbr,
        "tbr_err": tbr_err,
        "u238ng": u238ng,
        "Am241_fission": am241_fiss,
        "Am241_ngamma": am241_ng,
        "Am241_n2n": am241_n2n,
        "Am241_n3n": am241_n3n,
        "Am241_nXn": am241_nXn,
        "k_inf": k_inf,
        "k_inf_std": k_inf_std,
        "heating_W": heating_W,
        "fisq_W": fisq_W,
        "fusion_W": fusion_W,
    }


# ==============================================================================
# Export Am-241 data to CSV
# ==============================================================================
def export_am241_csv(uo2_data, uzr_data, figures_dir):
    """Write Am-241 reaction rates for both fuel types to a CSV."""
    out_csv = os.path.join(figures_dir, "am241_data.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shell_midpoint_cm",
            "UO2_Am241_fission",  "UO2_Am241_ngamma",
            "UO2_Am241_n2n",      "UO2_Am241_n3n",      "UO2_Am241_nXn",
            "UZr_Am241_fission",  "UZr_Am241_ngamma",
            "UZr_Am241_n2n",      "UZr_Am241_n3n",      "UZr_Am241_nXn",
        ])
        for i in range(len(uo2_data["midpoints"])):
            writer.writerow([
                f"{uo2_data['midpoints'][i]:.1f}",
                f"{uo2_data['Am241_fission'][i]:.6e}",
                f"{uo2_data['Am241_ngamma'][i]:.6e}",
                f"{uo2_data['Am241_n2n'][i]:.6e}",
                f"{uo2_data['Am241_n3n'][i]:.6e}",
                f"{uo2_data['Am241_nXn'][i]:.6e}",
                f"{uzr_data['Am241_fission'][i]:.6e}",
                f"{uzr_data['Am241_ngamma'][i]:.6e}",
                f"{uzr_data['Am241_n2n'][i]:.6e}",
                f"{uzr_data['Am241_n3n'][i]:.6e}",
                f"{uzr_data['Am241_nXn'][i]:.6e}",
            ])
    print(f"Saved: {out_csv}")


def export_pu239_csv(uo2_data, uzr_data, figures_dir):
    """Write Pu-239 production rates [kg/yr] for both fuel types to a CSV."""
    conv = S_NEUTRON * (M_PU239 / N_A) * SEC_PER_YR / 1000.0
    out_csv = os.path.join(figures_dir, "pu239_data.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shell_midpoint_cm",
            "UO2_U238_ngamma",  "UO2_Pu239_kg_per_yr",
            "UZr_U238_ngamma",  "UZr_Pu239_kg_per_yr",
        ])
        for i in range(len(uo2_data["midpoints"])):
            writer.writerow([
                f"{uo2_data['midpoints'][i]:.1f}",
                f"{uo2_data['u238ng'][i]:.6e}",
                f"{uo2_data['u238ng'][i] * conv:.4f}",
                f"{uzr_data['u238ng'][i]:.6e}",
                f"{uzr_data['u238ng'][i] * conv:.4f}",
            ])
    print(f"Saved: {out_csv}")


# ==============================================================================
# Plot
# ==============================================================================
def main():
    # --- Load both datasets ---
    have_uo2 = os.path.isfile(UO2_JSON)
    have_uzr = os.path.isfile(UZR_JSON)

    if not have_uo2 and not have_uzr:
        print("ERROR: Neither result JSON was found.")
        print(f"  Looked for: {UO2_JSON}")
        print(f"              {UZR_JSON}")
        print("Run uo2_zpinch.py and/or uzr_zpinch.py first.")
        return

    # Pre-load data so both figures can share it
    FIGURES_DIR = "figures"
    os.makedirs(FIGURES_DIR, exist_ok=True)

    uo2_data = None
    uzr_data = None
    if have_uo2:
        uo2_data = load_series(UO2_JSON)
        print(f"UO₂ loaded: {len(uo2_data['midpoints'])} positions")
    if have_uzr:
        uzr_data = load_series(UZR_JSON)
        print(f"U-10Zr loaded: {len(uzr_data['midpoints'])} positions")

    # --- Export CSVs ---
    if uo2_data is not None and uzr_data is not None:
        export_am241_csv(uo2_data, uzr_data, FIGURES_DIR)
        export_pu239_csv(uo2_data, uzr_data, FIGURES_DIR)

    # ==================================================================
    # Figure 1 — TBR vs blanket position
    # ==================================================================
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        ax1.plot(uo2_data["midpoints"], uo2_data["tbr"], "o-",
                 color="tab:blue", label=UO2_LABEL,
                 linewidth=0.75, markersize=3)

    if uzr_data is not None:
        ax1.plot(uzr_data["midpoints"], uzr_data["tbr"], "s--",
                 color="tab:red", label=UZR_LABEL,
                 linewidth=0.75, markersize=3)

    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel("Tritium breeding ratio (TBR)")
    ax1.legend(loc="best")
    ax1.grid(True)
    ax1.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out1 = os.path.join(FIGURES_DIR, f"tbr.{ext}")
        fig1.savefig(out1, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out1}")
    plt.close(fig1)

    # ==================================================================
    # Figure 2 — U238 (n,gamma) vs blanket position
    # ==================================================================
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        ax2.plot(uo2_data["midpoints"], uo2_data["u238ng"], "o-",
                 color="tab:blue", label=UO2_LABEL,
                 linewidth=0.75, markersize=3)

    if uzr_data is not None:
        ax2.plot(uzr_data["midpoints"], uzr_data["u238ng"], "s--",
                 color="tab:red", label=UZR_LABEL,
                 linewidth=0.75, markersize=3)

    ax2.set_xlabel(XLABEL)
    ax2.set_ylabel(r"U-238 (n,$\gamma$) [rxns/src-n]")
    ax2.legend(loc="best")
    ax2.grid(True)
    ax2.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out2 = os.path.join(FIGURES_DIR, f"u238ng.{ext}")
        fig2.savefig(out2, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out2}")
    plt.close(fig2)

    # ==================================================================
    # Figure 3 — Pu-239 production rate [kg/yr] vs blanket position
    # ==================================================================
    conv = S_NEUTRON * (M_PU239 / N_A) * SEC_PER_YR / 1000.0

    fig3, ax3 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        ax3.plot(uo2_data["midpoints"], uo2_data["u238ng"] * conv, "o-",
                 color="tab:blue", label=UO2_LABEL,
                 linewidth=0.75, markersize=3)

    if uzr_data is not None:
        ax3.plot(uzr_data["midpoints"], uzr_data["u238ng"] * conv, "s--",
                 color="tab:red", label=UZR_LABEL,
                 linewidth=0.75, markersize=3)

    ax3.set_xlabel(XLABEL)
    ax3.set_ylabel(r"Pu-239 production rate [kg/yr]")
    ax3.legend(loc="best")
    ax3.grid(True)
    ax3.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out3 = os.path.join(FIGURES_DIR, f"pu239.{ext}")
        fig3.savefig(out3, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out3}")
    plt.close(fig3)

    # ==================================================================
    # Figure 4 — Am-241 fission, (n,gamma), and (n,Xn) vs blanket position
    # ==================================================================
    fig4, ax4 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        mid = uo2_data["midpoints"]
        ax4.plot(mid, uo2_data["Am241_ngamma"], "^-",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UO2_LABEL + r" - Am241 (n,$\gamma$)")
        ax4.plot(mid, uo2_data["Am241_fission"], "o-",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 label=UO2_LABEL + r" - Am241 (n,fis)")
        ax4.plot(mid, uo2_data["Am241_nXn"], "o--",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 label=UO2_LABEL + r" - Am241 (n,Xn)")

    if uzr_data is not None:
        mid = uzr_data["midpoints"]
        ax4.plot(mid, uzr_data["Am241_ngamma"], "D-",
                 color="tab:red", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UZR_LABEL + r" - Am241 (n,$\gamma$)")
        ax4.plot(mid, uzr_data["Am241_fission"], "s-",
                 color="tab:red", linewidth=0.75, markersize=3,
                 label=UZR_LABEL + r" - Am241 (n,fis)")
        ax4.plot(mid, uzr_data["Am241_nXn"], "s--",
                 color="tab:red", linewidth=0.75, markersize=3,
                 label=UZR_LABEL + r" - Am241 (n,Xn)")

    ax4.set_xlabel(XLABEL)
    ax4.set_ylabel(r"Am-241 reaction rate [rxns/src-n]")
    ax4.set_yscale("log")
    ax4.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax4.legend(loc="best", ncol=2, fontsize=6)
    ax4.grid(True)
    ax4.set_xlim(-10, 210)
    ax4.set_ylim(1e-13, 1e-1)

    for ext in ("png", "pdf"):
        out4 = os.path.join(FIGURES_DIR, f"am241.{ext}")
        fig4.savefig(out4, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out4}")
    plt.close(fig4)

    # ==================================================================
    # Figure 5 — Spent fuel mass vs blanket position
    # ==================================================================
    def fuel_mass_kg(midpoints, rho_fuel):
        """Spent fuel mass [kg] in the full cylinder for each midpoint."""
        r_in  = R_INNER + midpoints - FUEL_SHELL_THICKNESS / 2.0
        r_out = R_INNER + midpoints + FUEL_SHELL_THICKNESS / 2.0
        vol   = np.pi * (r_out**2 - r_in**2) * BLANKET_HEIGHT   # cm³
        return vol * FUEL_VOL_FRAC * rho_fuel / 1000.0           # g → kg

    fig5, ax5 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        mid = uo2_data["midpoints"]
        ax5.plot(mid, fuel_mass_kg(mid, RHO_UO2), "o-",
                 color="tab:blue", label=UO2_LABEL,
                 linewidth=0.75, markersize=3)

    if uzr_data is not None:
        mid = uzr_data["midpoints"]
        ax5.plot(mid, fuel_mass_kg(mid, RHO_UZR), "s--",
                 color="tab:red", label=UZR_LABEL,
                 linewidth=0.75, markersize=3)

    ax5.set_xlabel(XLABEL)
    ax5.set_ylabel("Spent fuel mass [kg]")
    ax5.legend(loc="best")
    ax5.grid(True)
    ax5.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out5 = os.path.join(FIGURES_DIR, f"fuel_mass.{ext}")
        fig5.savefig(out5, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out5}")
    plt.close(fig5)

    # ==================================================================
    # Figure 6 — k-inf of fuel zone vs blanket position
    # ==================================================================
    fig6, ax6 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    if uo2_data is not None:
        mid = uo2_data["midpoints"]
        ax6.plot(mid, uo2_data["k_inf"], "o-",
                 color="tab:blue", label=UO2_LABEL,
                 linewidth=0.75, markersize=3)

    if uzr_data is not None:
        mid = uzr_data["midpoints"]
        ax6.plot(mid, uzr_data["k_inf"], "s--",
                 color="tab:red", label=UZR_LABEL,
                 linewidth=0.75, markersize=3)

    ax6.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax6.set_xlabel(XLABEL)
    ax6.set_ylabel(r"k-inf (nu-fis/abs) in spent fuel zone")
    ax6.legend(loc="center right")
    ax6.grid(True)
    ax6.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out6 = os.path.join(FIGURES_DIR, f"kinf.{ext}")
        fig6.savefig(out6, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out6}")
    plt.close(fig6)

    # ==================================================================
    # Figure 7 — Thermal power breakdown vs blanket position
    # ==================================================================
    fig7, ax7 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    W_TO_MW = 1.0e-6

    if uo2_data is not None:
        mid = uo2_data["midpoints"]
        ax7.plot(mid, uo2_data["heating_W"] * W_TO_MW, "o-",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 label=UO2_LABEL + " — Total")
        ax7.plot(mid, uo2_data["fisq_W"] * W_TO_MW, "^-",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UO2_LABEL + " — Fission")
        ax7.plot(mid, uo2_data["fusion_W"] * W_TO_MW, "v--",
                 color="tab:blue", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UO2_LABEL + " — Fusion")

    if uzr_data is not None:
        mid = uzr_data["midpoints"]
        ax7.plot(mid, uzr_data["heating_W"] * W_TO_MW, "s-",
                 color="tab:red", linewidth=0.75, markersize=3,
                 label=UZR_LABEL + " — Total")
        ax7.plot(mid, uzr_data["fisq_W"] * W_TO_MW, "D-",
                 color="tab:red", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UZR_LABEL + " — Fission")
        ax7.plot(mid, uzr_data["fusion_W"] * W_TO_MW, "P--",
                 color="tab:red", linewidth=0.75, markersize=3,
                 markerfacecolor="none",
                 label=UZR_LABEL + " — Fusion")

    ax7.set_xlabel(XLABEL)
    ax7.set_ylabel("Thermal power [MW]")
    ax7.legend(loc="best", ncol=2, fontsize=5.5)
    ax7.grid(True)
    ax7.set_xlim(-10, 210)

    for ext in ("png", "pdf"):
        out7 = os.path.join(FIGURES_DIR, f"heating.{ext}")
        fig7.savefig(out7, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved: {out7}")
    plt.close(fig7)


if __name__ == "__main__":
    main()