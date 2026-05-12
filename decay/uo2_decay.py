"""
OpenMC Decay-Only Calculation — AP1000 UO₂ Spent Fuel in Storage
================================================================

Reads an end-of-life isotopic inventory (atoms) from a CSV produced by
uo2_deplete.py and decays it in-place for 73 years with no neutron
flux — pure radioactive decay via the Bateman equations solved by
OpenMC's CRAM-48 solver.

73 years of cooling maximises the Am-241 inventory from Pu-241 beta
decay (t½ = 14.3 yr → Am-241, t½ = 432 yr).  After ~5 Pu-241
half-lives essentially all Pu-241 has converted, and Am-241 has
barely decayed.  This represents a near-worst-case (or best-case,
depending on your perspective) minor-actinide vector for spent PWR
fuel as potential feedstock for a Z-pinch fusion–fission hybrid
blanket.

The script is split into three functions so that individual stages
can be skipped (e.g. if the HDF5 results already exist from a
previous run):

    build_operator()  — read CSV, set up IndependentOperator
    run_decay()       — integrate and write depletion_results.h5
    post_process()    — extract results, print summary, write CSV

Adjust parameters in the "USER-EDITABLE PARAMETERS" section as needed.
"""

import os
import csv
import numpy as np
import openmc
import openmc.deplete

# ============================================================
# USER-EDITABLE PARAMETERS
# ============================================================

# --- Input inventory ---
INPUT_CSV  = "uo2_eol_inventory_tracked.csv"
OUTPUT_CSV = "uo2_bol_inventory_zpinch.csv"

# --- Decay schedule ---
DECAY_YEARS   = 73.0                           # peak Am-241 buildup
DAYS_PER_YEAR = 365.25
DECAY_DAYS    = DECAY_YEARS * DAYS_PER_YEAR     # ≈ 26,663 days
NUM_STEPS     = 73                              # one step per year

# --- Depletion chain ---
# Use a PWR / thermal chain for UO₂ spent fuel.
# Pre-built chains available at https://openmc.org/depletion-chains/
openmc.config['chain_file'] = "/home/patri/openmc/data/chain_endfb81_thermal.xml"

# --- Arbitrary volume for bookkeeping ---
# With V = 1 cm³ the number density numerically equals the atom count.
VOLUME_CM3 = 1.0   # [cm³]

# --- Minimum atom count for Z-pinch output CSV ---
# Nuclides below this threshold are dropped from OUTPUT_CSV.
# 1e14 keeps all minor actinides; 1e16 would clip Cm243-245 and Am243.
MIN_ATOMS = 1.0e14

# --- Output directories ---
OPENMC_DIR  = "uo2_openmc"   # XMLs, H5s, all OpenMC working files
RESULTS_DIR = "./"           # final CSVs, plots

# --- Output file ---
DEPLETE_FILE = os.path.join(OPENMC_DIR, "depletion_results.h5")


# ============================================================
# FUNCTION 1: BUILD OPERATOR
# ============================================================

def build_operator():
    """Read the EOL inventory CSV and build a decay-only IndependentOperator.

    Returns
    -------
    op : openmc.deplete.IndependentOperator
        Transport-independent operator with empty MicroXS (decay only).
    nuclide_atoms : dict
        Original {nuclide: atoms} inventory as read from the CSV.
    """

    # --- Read the EOL isotopic inventory ---
    nuclide_atoms = {}
    with open(INPUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nuclide_atoms[row['nuclide']] = float(row['atoms'])

    total_atoms = sum(nuclide_atoms.values())
    print(f"Loaded {len(nuclide_atoms)} nuclides from {INPUT_CSV}")
    print(f"  Total atoms: {total_atoms:.6e}")

    # --- Convert absolute atoms → number densities [atom/cm³] ---
    #     n_i = N_i / V.  With V = 1 cm³ they are numerically equal.
    nuclide_densities = {nuc: atoms / VOLUME_CM3
                         for nuc, atoms in nuclide_atoms.items()}

    # --- Build the decay-only IndependentOperator ---
    #     MicroXS is populated with all-zero cross sections.  Combined
    #     with zero_flux the product is zero for every reaction, so only
    #     radioactive decay terms survive in the Bateman matrix.
    nuc_list = list(nuclide_densities.keys())
    micro_xs = openmc.deplete.MicroXS(
        data=np.zeros((len(nuc_list), 1, 1)),
        nuclides=nuc_list,
        reactions=['fission'],
    )
    zero_flux = np.array([0.0])

    op = openmc.deplete.IndependentOperator.from_nuclides(
        volume=VOLUME_CM3,
        nuclides=nuclide_densities,
        flux=zero_flux,
        micro_xs=micro_xs,
        nuc_units='atom/cm3',
        normalization_mode='source-rate',
    )

    return op, nuclide_atoms


# ============================================================
# FUNCTION 2: RUN DECAY
# ============================================================

def run_decay(op):
    """Set up the integrator and run the decay calculation.

    All OpenMC intermediate files (HDF5, etc.) are written into
    OPENMC_DIR so they don't clutter the working directory.

    Parameters
    ----------
    op : openmc.deplete.IndependentOperator
        The decay-only operator returned by build_operator().
    """

    print("=" * 60)
    print("AP1000 UO₂ Spent-Fuel Decay — 73 yr (peak Am-241)")
    print("=" * 60)
    print(f"  Input         : {INPUT_CSV}")
    print(f"  Decay time    : {DECAY_YEARS:.0f} years ({DECAY_DAYS:.0f} days)")
    print(f"  Time steps    : {NUM_STEPS}  ({DECAY_DAYS/NUM_STEPS:.1f} days/step)")
    print(f"  Output dir    : {OPENMC_DIR}/")
    print("=" * 60)

    dt           = DECAY_DAYS / NUM_STEPS
    timesteps    = [dt] * NUM_STEPS
    source_rates = [0.0] * NUM_STEPS        # zero flux each step → pure decay

    integrator = openmc.deplete.PredictorIntegrator(
        operator=op,
        timesteps=timesteps,
        source_rates=source_rates,
        timestep_units='d',
    )

    # Run from OPENMC_DIR so all intermediate files land there.
    orig_dir = os.getcwd()
    os.chdir(OPENMC_DIR)

    integrator.integrate()

    os.chdir(orig_dir)
    print("\nDecay calculation complete.")


# ============================================================
# FUNCTION 3: POST-PROCESS DECAY RESULTS
# ============================================================

def post_process(nuclide_atoms):
    """Read depletion_results.h5, print a summary, and write output CSVs.

    Parameters
    ----------
    nuclide_atoms : dict
        Original {nuclide: atoms} inventory (for comparison / bookkeeping).
    """

    results = openmc.deplete.Results(DEPLETE_FILE)

    # --- Material ID (auto-assigned by from_nuclides, usually "1") ---
    r0      = results[0]
    mat_id  = list(r0.index_mat.keys())[0]
    mat_idx = r0.index_mat[mat_id]

    # --- Gather initial and final atom counts for every nuclide ---
    # StepResult.data[mat, nuc] is already in absolute atoms.
    last = results[-1]
    all_nucs = set(r0.index_nuc.keys()) | set(last.index_nuc.keys())

    rows = []
    for nuc in sorted(all_nucs):
        atoms_i = 0.0
        atoms_f = 0.0
        if nuc in r0.index_nuc:
            atoms_i = r0.data[mat_idx, r0.index_nuc[nuc]]
        if nuc in last.index_nuc:
            atoms_f = last.data[mat_idx, last.index_nuc[nuc]]
        rows.append({
            'nuclide':       nuc,
            'atoms_initial': atoms_i,
            'atoms_final':   atoms_f,
            'delta_atoms':   atoms_f - atoms_i,
        })

    # --- Console summary: top 30 movers ---
    print(f"\n{'='*72}")
    print(f" Decay summary: {DECAY_YEARS:.0f} years in spent-fuel storage")
    print(f"{'='*72}")
    print(f"{'Nuclide':<12} {'Initial atoms':>16} {'Final atoms':>16} {'Change %':>10}")
    print(f"{'-'*12} {'-'*16} {'-'*16} {'-'*10}")

    rows_sorted = sorted(rows, key=lambda r: abs(r['delta_atoms']), reverse=True)
    for r in rows_sorted[:30]:
        ni = r['atoms_initial']
        nf = r['atoms_final']
        if ni > 0:
            pct = 100.0 * (nf - ni) / ni
            pct_str = f"{pct:+.4f}%"
        elif nf > 0:
            pct_str = "new"
        else:
            continue
        print(f"{r['nuclide']:<12} {ni:>16.6e} {nf:>16.6e} {pct_str:>10}")

    # --- Write full inventory CSV (initial + final + delta) ---
    csv_full = os.path.join(RESULTS_DIR, "uo2_delta_inventory.csv")
    with open(csv_full, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'nuclide', 'atoms_initial', 'atoms_final', 'delta_atoms'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Full inventory saved to {csv_full}")

    # --- Write final-state-only CSV (same format as input, for chaining) ---
    # Drop nuclides below MIN_ATOMS — removes numerical noise and
    # negligible daughters while keeping all minor actinides.
    csv_final = os.path.join(RESULTS_DIR, OUTPUT_CSV)
    n_kept = 0
    n_dropped = 0
    with open(csv_final, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['nuclide', 'atoms'])
        for r in rows:
            if r['atoms_final'] >= MIN_ATOMS:
                writer.writerow([r['nuclide'], f"{r['atoms_final']:.6e}"])
                n_kept += 1
            elif r['atoms_final'] > 0:
                n_dropped += 1
    print(f"  Final-only inventory saved to {csv_final}"
          f"  ({n_kept} nuclides kept, {n_dropped} dropped below {MIN_ATOMS:.0e})")

    # --- Print tracked nuclides of interest ---
    # Must match nuclides_tracked in uo2_deplete.py so the full
    # Z-pinch blanket input vector is visible here.
    tracked = [
        # Actinides
        "U234", "U235", "U236", "U238",
        "Np237",
        "Pu238", "Pu239", "Pu240", "Pu241", "Pu242",
        "Am241", "Am242_m1", "Am243",
        "Cm242", "Cm243", "Cm244", "Cm245",
        # High-absorption fission products
        "Sm149", "Sm151", "Gd155", "Gd157",
        "Rh103", "Nd143", "Nd145", "Cs133",
        "Eu151", "Eu153", "Ag109",
        # High-inventory fission products
        "Mo95", "Mo97", "Mo98", "Mo100",
        "Xe126", "Xe128", "Xe130", "Xe131", "Xe132", "Xe134", "Xe135", "Xe136",
        "Kr80", "Kr82", "Kr83", "Kr84", "Kr86",
        "Cs135", "Cs137",
        "Ba138", "La139", "Ce140", "Ce142",
        "Tc99", "Pr141", "Ru101", "Ru102",
        "Nd144", "Nd146", "Y89", "Sr88", "Sr90",
    ]

    print(f"\n  Tracked nuclide comparison (initial → final):")
    print(f"  {'-'*60}")
    for nuc in tracked:
        if nuc in last.index_nuc:
            nf = last.data[mat_idx, last.index_nuc[nuc]]
            ni = nuclide_atoms.get(nuc, 0.0)
            if ni > 0:
                pct = 100.0 * (nf - ni) / ni
                print(f"  {nuc:10s}  {ni:>14.6e} → {nf:>14.6e}  ({pct:+.2f}%)")
            elif nf > 0:
                print(f"  {nuc:10s}  {'—':>14s} → {nf:>14.6e}  (daughter)")
        else:
            print(f"  {nuc:10s}  (not in chain)")

    print(f"\n  OpenMC HDF5 results → {DEPLETE_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Create output directories
    os.makedirs(OPENMC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Always build the operator (fast, and we need nuclide_atoms for post-processing)
    op, nuclide_atoms = build_operator()

    if os.path.isfile(DEPLETE_FILE):
        print(f"\n{DEPLETE_FILE} already exists — skipping decay, "
              f"jumping to post-processing.\n")
    else:
        run_decay(op)

    post_process(nuclide_atoms)