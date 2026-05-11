#!/usr/bin/env python3
"""
==============================================================================
Zap Energy Fusion-Fission Hybrid Reactor — OpenMC Model
==============================================================================

Geometry: 2D R-Z cylindrical model (axially symmetric)
    - Central vacuum channel (plasma/Z-pinch source)
    - Inner Pb-Li gap (vortex flow clearance)
    - Spent fuel annulus (homogenized hex-can subassemblies from 4S reactor)
    - Outer Pb-Li breeding & shielding zone
    - Steel vessel wall
    - B4C + steel shield

Source: 14.1 MeV D-T neutrons, uniform line source along the central axis

Fuel: Spent U-10Zr from a 4S-type fast reactor, ~34,000 MWd/t burnup
      (approximate discharge isotopics for fast-spectrum metallic fuel)

References:
    - Zap Energy, "Integrated Approach to Fission and Fusion" (April 2026)
    - Ma et al., "Neutronic analysis of Z-FFR blanket," HPLPB 27 (2015)
    - Meyer et al., "The EBR-II X501 minor actinide burning experiment," JNM (2009)
    - Toshiba 4S design: U-10Zr, HT-9 cladding, 17-19% enriched, 30-year core life

Author: Generated for AST 570 project
==============================================================================
"""

import openmc
import numpy as np


# ==========================================================================
# 1. MATERIALS
# ==========================================================================

# --- 1a. Pb-Li eutectic (Pb-83at%Li-17at%, also written Pb-15.7wt%Li) ---
# This is the baseline blanket fluid: coolant, tritium breeder, neutron multiplier
pbli = openmc.Material(name="PbLi_eutectic")
# Atomic fractions: 83% Pb-nat, 17% Li (natural: 7.5% Li-6, 92.5% Li-7)
pbli.add_element("Pb", 0.83, percent_type="ao")
pbli.add_nuclide("Li6", 0.17 * 0.075, percent_type="ao")   # 7.5% of Li is Li-6
pbli.add_nuclide("Li7", 0.17 * 0.925, percent_type="ao")   # 92.5% of Li is Li-7
pbli.set_density("g/cm3", 9.4)

# --- 1b. Spent U-10Zr fuel (discharged from 4S-type fast reactor) ---
# Approximate isotopics after ~34,000 MWd/t burnup in a fast spectrum.
# Starting composition: U-235 18%, U-238 72%, Zr 10% (by weight).
# After 30 years at low power density, significant Am-241 from Pu-241 decay.
#
# These are APPROXIMATE mass fractions — a proper ORIGEN depletion of the
# 4S pin cell should be run to get precise values.
spent_fuel = openmc.Material(name="Spent_U10Zr_4S")
# Uranium isotopes
spent_fuel.add_nuclide("U235",  0.1100, percent_type="wo")  # depleted from 18%
spent_fuel.add_nuclide("U238",  0.6800, percent_type="wo")  # still dominant
# Plutonium isotopes (bred from U-238 capture)
spent_fuel.add_nuclide("Pu239", 0.0420, percent_type="wo")
spent_fuel.add_nuclide("Pu240", 0.0150, percent_type="wo")
spent_fuel.add_nuclide("Pu241", 0.0030, percent_type="wo")
spent_fuel.add_nuclide("Pu242", 0.0010, percent_type="wo")
# Minor actinides
spent_fuel.add_nuclide("Am241", 0.0035, percent_type="wo")  # from Pu-241 decay over 30yr
spent_fuel.add_nuclide("Np237", 0.0015, percent_type="wo")
spent_fuel.add_nuclide("Cm244", 0.0005, percent_type="wo")
# Zirconium (alloying element, unchanged)
spent_fuel.add_element("Zr",    0.1000, percent_type="wo")
# Representative fission products (lumped — in a real model, include
# the top ~20 FPs individually from ORIGEN output)
spent_fuel.add_nuclide("Mo95",  0.0050, percent_type="wo")
spent_fuel.add_nuclide("Cs133", 0.0030, percent_type="wo")
spent_fuel.add_nuclide("Nd143", 0.0025, percent_type="wo")
spent_fuel.add_nuclide("Ru101", 0.0020, percent_type="wo")
spent_fuel.add_nuclide("Pd105", 0.0015, percent_type="wo")
spent_fuel.add_nuclide("Ce140", 0.0020, percent_type="wo")
spent_fuel.add_nuclide("Zr93",  0.0020, percent_type="wo")
spent_fuel.add_nuclide("Sr90",  0.0010, percent_type="wo")
spent_fuel.add_nuclide("Tc99",  0.0010, percent_type="wo")
spent_fuel.add_nuclide("Cs137", 0.0010, percent_type="wo")
# Remaining mass assigned to Ba (catch-all for other FPs)
remaining = 1.0 - (0.11 + 0.68 + 0.042 + 0.015 + 0.003 + 0.001
                    + 0.0035 + 0.0015 + 0.0005 + 0.10
                    + 0.005 + 0.003 + 0.0025 + 0.002 + 0.0015
                    + 0.002 + 0.002 + 0.001 + 0.001 + 0.001)
spent_fuel.add_nuclide("Ba138", max(remaining, 0.0), percent_type="wo")
# Metallic fuel density (U-10Zr alloy with some swelling/porosity, ~75% smear)
# Theoretical density ~15.8 g/cm3, at 75% smear density: ~11.9 g/cm3
# After burnup swelling, effective ~11.0 g/cm3
spent_fuel.set_density("g/cm3", 11.0)

# --- 1c. HT-9 ferritic-martensitic steel (cladding & hex duct) ---
ht9 = openmc.Material(name="HT9_steel")
ht9.add_element("Fe", 0.845, percent_type="wo")
ht9.add_element("Cr", 0.120, percent_type="wo")
ht9.add_element("Mo", 0.010, percent_type="wo")
ht9.add_element("W",  0.005, percent_type="wo")
ht9.add_element("V",  0.003, percent_type="wo")
ht9.add_element("Mn", 0.006, percent_type="wo")
ht9.add_element("Ni", 0.005, percent_type="wo")
ht9.add_element("Si", 0.004, percent_type="wo")
ht9.add_element("C",  0.002, percent_type="wo")
ht9.set_density("g/cm3", 7.87)

# --- 1d. SS316 stainless steel (vessel wall) ---
ss316 = openmc.Material(name="SS316")
ss316.add_element("Fe", 0.665, percent_type="wo")
ss316.add_element("Cr", 0.170, percent_type="wo")
ss316.add_element("Ni", 0.120, percent_type="wo")
ss316.add_element("Mo", 0.025, percent_type="wo")
ss316.add_element("Mn", 0.020, percent_type="wo")
ss316.set_density("g/cm3", 7.99)

# --- 1e. B4C shield ---
b4c = openmc.Material(name="B4C_shield")
b4c.add_element("B", 4, percent_type="ao")
b4c.add_element("C", 1, percent_type="ao")
b4c.set_density("g/cm3", 2.52)

# --- 1f. Homogenized fuel zone ---
# In the annular fuel zone, hex-can subassemblies sit in flowing Pb-Li.
# Volume fractions (approximate for wire-wrapped pin bundle in hex duct):
#   Fuel slugs:    ~30%  (metallic fuel at 11.0 g/cm3)
#   HT-9 steel:    ~20%  (cladding + wire wrap + hex duct wall)
#   Pb-Li coolant: ~50%  (flowing through and around the assemblies)
#
# For a first-pass scoping model, we homogenize this into one material.
# A refined model would resolve individual pin cells.
fuel_zone_homog = openmc.Material.mix_materials(
    [spent_fuel, ht9, pbli],
    [0.30, 0.20, 0.50],
    percent_type="vo",
    name="Fuel_zone_homogenized"
)

# --- Collect all materials ---
materials = openmc.Materials([pbli, spent_fuel, ht9, ss316, b4c, fuel_zone_homog])
materials.export_to_xml()


# ==========================================================================
# 2. GEOMETRY
# ==========================================================================
# Cylindrical R-Z model, axially symmetric about the Z-axis.
#
# Radial zones (from center outward):
#   r = 0   - 15 cm : Vacuum (plasma channel / Z-pinch)
#   r = 15  - 25 cm : Pb-Li (inner gap, vortex flow clearance)
#   r = 25  - 45 cm : Homogenized fuel zone (spent U-10Zr hex cans in Pb-Li)
#   r = 45  - 190 cm: Pb-Li (outer breeding & shielding)
#   r = 190 - 195 cm: SS316 vessel wall
#   r = 195 - 210 cm: B4C + steel shield
#
# Axial extent: z = -150 to +150 cm (total height 300 cm = 3 m)

# Radial surfaces (ZCylinder)
r1 = openmc.ZCylinder(r=15.0)    # plasma channel outer wall
r2 = openmc.ZCylinder(r=25.0)    # inner edge of fuel zone
r3 = openmc.ZCylinder(r=45.0)    # outer edge of fuel zone
r4 = openmc.ZCylinder(r=190.0)   # vessel inner wall
r5 = openmc.ZCylinder(r=195.0)   # vessel outer wall
r6 = openmc.ZCylinder(r=210.0, boundary_type="vacuum")  # shield outer

# Axial surfaces (ZPlane)
z_bot = openmc.ZPlane(z0=-150.0, boundary_type="vacuum")
z_top = openmc.ZPlane(z0=+150.0, boundary_type="vacuum")

# Define cells
# Zone 1: Vacuum (plasma channel)
cell_plasma = openmc.Cell(name="plasma_channel")
cell_plasma.region = -r1 & +z_bot & -z_top
# No material assigned — void/vacuum

# Zone 2: Inner Pb-Li gap
cell_inner_pbli = openmc.Cell(name="inner_PbLi")
cell_inner_pbli.fill = pbli
cell_inner_pbli.region = +r1 & -r2 & +z_bot & -z_top

# Zone 3: Fuel zone (homogenized hex cans + Pb-Li)
cell_fuel = openmc.Cell(name="fuel_zone")
cell_fuel.fill = fuel_zone_homog
cell_fuel.region = +r2 & -r3 & +z_bot & -z_top

# Zone 4: Outer Pb-Li breeding zone
cell_outer_pbli = openmc.Cell(name="outer_PbLi")
cell_outer_pbli.fill = pbli
cell_outer_pbli.region = +r3 & -r4 & +z_bot & -z_top

# Zone 5: Steel vessel
cell_vessel = openmc.Cell(name="vessel_wall")
cell_vessel.fill = ss316
cell_vessel.region = +r4 & -r5 & +z_bot & -z_top

# Zone 6: Shield
cell_shield = openmc.Cell(name="shield")
cell_shield.fill = b4c
cell_shield.region = +r5 & -r6 & +z_bot & -z_top

# Root universe
root_universe = openmc.Universe(cells=[
    cell_plasma, cell_inner_pbli, cell_fuel,
    cell_outer_pbli, cell_vessel, cell_shield
])
geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()


# ==========================================================================
# 3. SOURCE DEFINITION
# ==========================================================================
# 14.1 MeV D-T fusion neutrons emitted isotropically from a line source
# along the Z-axis (r = 0), uniformly distributed from z = -100 to +100 cm.
# (Active plasma length shorter than full tank height)

source = openmc.IndependentSource()

# Spatial: line source along Z-axis
source.space = openmc.stats.CylindricalIndependent(
    r=openmc.stats.Discrete([0.0], [1.0]),         # r = 0 (on axis)
    phi=openmc.stats.Uniform(0.0, 2 * np.pi),      # uniform in phi
    z=openmc.stats.Uniform(-100.0, 100.0),          # active plasma region
    origin=(0.0, 0.0, 0.0)
)

# Angular: isotropic
source.angle = openmc.stats.Isotropic()

# Energy: monoenergetic 14.1 MeV (D-T fusion neutron)
source.energy = openmc.stats.Discrete([14.1e6], [1.0])

source.particle = "neutron"


# ==========================================================================
# 4. SETTINGS
# ==========================================================================

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = [source]
settings.batches = 100
settings.particles = 100000       # 100k per batch = 10M total histories
settings.output = {"tallies": True}

# Photon transport (important for energy deposition accuracy, as noted
# in the Ma et al. paper — photon contribution ~5% to energy deposition)
settings.photon_transport = True

settings.export_to_xml()


# ==========================================================================
# 5. TALLIES
# ==========================================================================
tallies = openmc.Tallies()

# --- 5a. Cell filters ---
filter_fuel     = openmc.CellFilter([cell_fuel])
filter_inner_li = openmc.CellFilter([cell_inner_pbli])
filter_outer_li = openmc.CellFilter([cell_outer_pbli])
filter_all_li   = openmc.CellFilter([cell_inner_pbli, cell_outer_pbli])
filter_all      = openmc.CellFilter([
    cell_inner_pbli, cell_fuel, cell_outer_pbli, cell_vessel, cell_shield
])

# --- 5b. Energy filter for spectral tallies ---
energy_bins = np.logspace(-5, 7.3, 201)  # 10 meV to 20 MeV, 200 groups
filter_energy = openmc.EnergyFilter(energy_bins)

# --- 5c. Radial mesh for spatial profiles ---
mesh_r = openmc.CylindricalMesh(
    r_grid=np.linspace(0, 210, 211),           # 1 cm radial bins
    z_grid=[-150.0, 150.0],                    # full height (axially integrated)
    phi_grid=[0, 2 * np.pi]                    # full azimuth
)
filter_mesh = openmc.MeshFilter(mesh_r)


# ---- TALLY 1: Tritium Breeding Ratio (TBR) ----
# Tritium production from Li-6(n,t)He-4 [MT=205] and Li-7(n,n't)He-4 [MT=205]
# OpenMC scores '(n,Xt)' which counts all tritium-producing reactions
tally_tbr = openmc.Tally(name="TBR")
tally_tbr.filters = [filter_all_li]
tally_tbr.nuclides = ["Li6", "Li7"]
tally_tbr.scores = ["(n,Xt)"]
tallies.append(tally_tbr)

# Also tally tritium production in the fuel zone (Li in the Pb-Li coolant
# within the homogenized fuel region)
tally_tbr_fuel = openmc.Tally(name="TBR_fuel_zone")
tally_tbr_fuel.filters = [filter_fuel]
tally_tbr_fuel.scores = ["(n,Xt)"]
tallies.append(tally_tbr_fuel)


# ---- TALLY 2: Fission and multiplication in fuel zone ----
tally_fission = openmc.Tally(name="fuel_reactions")
tally_fission.filters = [filter_fuel]
tally_fission.nuclides = ["U235", "U238", "Pu239", "Pu240",
                           "Pu241", "Am241", "Np237", "Cm244"]
tally_fission.scores = [
    "fission",             # total fission rate
    "nu-fission",          # neutrons produced by fission (nu * fission)
    "(n,2n)",              # neutron multiplication
    "(n,3n)",              # neutron multiplication
    "(n,gamma)",           # radiative capture (parasitic for some, breeding for others)
]
tallies.append(tally_fission)


# ---- TALLY 3: Energy deposition (for M, the energy multiplication factor) ----
tally_heating = openmc.Tally(name="heating_by_zone")
tally_heating.filters = [filter_all]
tally_heating.scores = ["heating"]  # eV/source-particle deposited
tallies.append(tally_heating)


# ---- TALLY 4: Neutron flux spectrum in fuel zone ----
tally_spectrum = openmc.Tally(name="fuel_spectrum")
tally_spectrum.filters = [filter_fuel, filter_energy]
tally_spectrum.scores = ["flux"]
tallies.append(tally_spectrum)


# ---- TALLY 5: Radial flux and heating profile ----
tally_radial = openmc.Tally(name="radial_profiles")
tally_radial.filters = [filter_mesh]
tally_radial.scores = ["flux", "heating"]
tallies.append(tally_radial)


# ---- TALLY 6: Neutron balance (leakage, absorption, production) ----
tally_balance = openmc.Tally(name="neutron_balance")
tally_balance.filters = [filter_all]
tally_balance.scores = ["absorption", "nu-fission", "(n,2n)", "(n,3n)"]
tallies.append(tally_balance)


# ---- TALLY 7: Delayed vs prompt nu-fission (for pulse analysis) ----
tally_delayed = openmc.Tally(name="delayed_neutrons")
tally_delayed.filters = [filter_fuel]
tally_delayed.scores = ["prompt-nu-fission", "delayed-nu-fission"]
tallies.append(tally_delayed)


# ---- TALLY 8: Transmutation-specific reaction rates ----
# These are the reactions that destroy minor actinides
tally_transmute = openmc.Tally(name="transmutation_rates")
tally_transmute.filters = [filter_fuel]
tally_transmute.nuclides = ["Am241", "Np237", "Cm244", "Pu239", "Pu240"]
tally_transmute.scores = [
    "fission",      # destruction by fission
    "(n,2n)",       # transmutation to adjacent isotope
    "(n,3n)",       # transmutation
    "(n,gamma)",    # capture (may produce higher actinide)
    "absorption",   # total destruction rate
]
tallies.append(tally_transmute)


tallies.export_to_xml()


# ==========================================================================
# 6. GEOMETRY VISUALIZATION (optional, generates plots)
# ==========================================================================

# R-Z cross-section (X-Z plane at Y=0)
plot_rz = openmc.Plot()
plot_rz.filename = "rz_cross_section"
plot_rz.basis = "xz"
plot_rz.origin = (0, 0, 0)
plot_rz.width = (440, 320)
plot_rz.pixels = (2200, 1600)
plot_rz.color_by = "material"

# R-theta cross-section (X-Y plane at Z=0, midplane)
plot_xy = openmc.Plot()
plot_xy.filename = "xy_cross_section"
plot_xy.basis = "xy"
plot_xy.origin = (0, 0, 0)
plot_xy.width = (440, 440)
plot_xy.pixels = (2200, 2200)
plot_xy.color_by = "material"

plots = openmc.Plots([plot_rz, plot_xy])
plots.export_to_xml()


# ==========================================================================
# 7. POST-PROCESSING SCRIPT (run after simulation completes)
# ==========================================================================
#
# After running `openmc`, use the following to extract results:
#
# import openmc
# sp = openmc.StatePoint("statepoint.100.h5")
#
# # --- TBR ---
# tbr_tally = sp.get_tally(name="TBR")
# tbr_df = tbr_tally.get_pandas_dataframe()
# tbr_total = tbr_df["mean"].sum()
# tbr_unc = np.sqrt((tbr_df["std. dev."]**2).sum())
# print(f"TBR (Li zones only) = {tbr_total:.4f} +/- {tbr_unc:.4f}")
#
# tbr_fuel = sp.get_tally(name="TBR_fuel_zone")
# tbr_fuel_df = tbr_fuel.get_pandas_dataframe()
# tbr_fuel_total = tbr_fuel_df["mean"].sum()
# print(f"TBR (fuel zone Li)  = {tbr_fuel_total:.4f}")
# print(f"TBR (total)         = {tbr_total + tbr_fuel_total:.4f}")
#
# # --- Energy multiplication ---
# heat_tally = sp.get_tally(name="heating_by_zone")
# heat_df = heat_tally.get_pandas_dataframe()
# total_heat_eV = heat_df["mean"].sum()  # eV per source neutron
# E_fusion = 14.1e6  # eV per D-T neutron (only neutron energy; total is 17.6 MeV)
# M = total_heat_eV / E_fusion
# print(f"Energy multiplication M = {M:.2f}")
#
# # --- Fission rates ---
# fis_tally = sp.get_tally(name="fuel_reactions")
# fis_df = fis_tally.get_pandas_dataframe()
# print(fis_df.to_string())
#
# # --- Transmutation rates ---
# trans_tally = sp.get_tally(name="transmutation_rates")
# trans_df = trans_tally.get_pandas_dataframe()
# print(trans_df.to_string())
#
# # --- Radial profiles ---
# import matplotlib.pyplot as plt
# radial = sp.get_tally(name="radial_profiles")
# radial_df = radial.get_pandas_dataframe()
# # Extract flux and heating vs radial bin
# # (parse mesh filter indices to get r values)


print("=" * 70)
print("OpenMC model files generated:")
print("  materials.xml")
print("  geometry.xml")
print("  settings.xml")
print("  tallies.xml")
print("  plots.xml")
print()
print("To run:")
print("  1. Download cross-section data (ENDF/B-VIII.0 or FENDL-3.2)")
print("  2. Set OPENMC_CROSS_SECTIONS environment variable")
print("  3. Run: openmc")
print("  4. Run: openmc --plot  (for geometry visualization)")
print("  5. Post-process with the script at the bottom of this file")
print("=" * 70)