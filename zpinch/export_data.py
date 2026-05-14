#!/usr/bin/env python3
"""
Export Am-241 reaction-rate data (as plotted in Figure 4 of plot.py) to CSV.

Reads the same JSON + neutron-balance CSVs that plot.py uses and writes
a single CSV with columns for both fuel types.

Usage:
    python export_am241_csv.py

Output:
    figures/am241_data.csv
"""

import os, json, csv
import numpy as np

# --- Paths (same as plot.py) ---
UO2_JSON = "uo2_zpinch_results_shell20cm/uo2_zpinch_all_results.json"
UZR_JSON = "uzr_zpinch_results_shell20cm/uzr_zpinch_all_results.json"

OUT_CSV  = "figures/am241_data.csv"


def read_nbal_column_sum(nbal_path, col_name):
    """Sum a single column from a neutron-balance depth CSV."""
    if nbal_path is None or not os.path.isfile(nbal_path):
        return 0.0
    with open(nbal_path) as f:
        rows = list(csv.DictReader(f))
    return sum(float(r.get(col_name, 0.0)) for r in rows)


def extract_am241(json_path):
    """Return arrays of (midpoints, fission, ngamma, n2n, n3n, nXn)."""
    with open(json_path) as f:
        results = json.load(f)
    fuel = [r for r in results if not r.get("no_fuel", False)]

    mid  = np.array([r["shell_midpoint_cm"] for r in fuel])
    fis  = np.array([read_nbal_column_sum(r.get("nbal_csv"), "Am241_fission") for r in fuel])
    ng   = np.array([read_nbal_column_sum(r.get("nbal_csv"), "Am241_ngamma")  for r in fuel])
    n2n  = np.array([read_nbal_column_sum(r.get("nbal_csv"), "Am241_n2n")     for r in fuel])
    n3n  = np.array([read_nbal_column_sum(r.get("nbal_csv"), "Am241_n3n")     for r in fuel])

    return mid, fis, ng, n2n, n3n, n2n + n3n


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    uo2_mid, uo2_f, uo2_ng, uo2_n2n, uo2_n3n, uo2_nXn = extract_am241(UO2_JSON)
    uzr_mid, uzr_f, uzr_ng, uzr_n2n, uzr_n3n, uzr_nXn = extract_am241(UZR_JSON)

    # Verify midpoints match (they should — same sweep)
    assert np.allclose(uo2_mid, uzr_mid), "Midpoint arrays differ between fuel types"

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shell_midpoint_cm",
            "UO2_Am241_fission",  "UO2_Am241_ngamma",
            "UO2_Am241_n2n",      "UO2_Am241_n3n",      "UO2_Am241_nXn",
            "UZr_Am241_fission",  "UZr_Am241_ngamma",
            "UZr_Am241_n2n",      "UZr_Am241_n3n",      "UZr_Am241_nXn",
        ])
        for i in range(len(uo2_mid)):
            writer.writerow([
                f"{uo2_mid[i]:.1f}",
                f"{uo2_f[i]:.6e}",   f"{uo2_ng[i]:.6e}",
                f"{uo2_n2n[i]:.6e}", f"{uo2_n3n[i]:.6e}", f"{uo2_nXn[i]:.6e}",
                f"{uzr_f[i]:.6e}",   f"{uzr_ng[i]:.6e}",
                f"{uzr_n2n[i]:.6e}", f"{uzr_n3n[i]:.6e}", f"{uzr_nXn[i]:.6e}",
            ])

    print(f"Wrote {len(uo2_mid)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()