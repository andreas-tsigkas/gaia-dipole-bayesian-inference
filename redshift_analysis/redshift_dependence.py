"""
Redshift dependence analysis for the dipole model.

This script studies how the inferred dipole glide and rotation amplitudes
depend on redshift by splitting the Gaia DR3 QSO sample into redshift bins
and fitting the same Bayesian model independently in each bin.

The underlying statistical model is defined in model/dipole_model.py.
Only the data selection, binning, and posterior summarization are handled here.

The input DataFrame `df` is assumed to already exist and must contain:
ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, redshift_qsoc.
"""


import numpy as np
import pandas as pd
import arviz as az

from model.dipole_model import toroidal_dipole_model

df = df[df["redshift_qsoc"].notna()]

redshift_bins = [(0.023, 1.25), (1.25, 2.2), (2.2, 6)]
#redshift_bins = [(0.023, 1), (1, 1.65), (1.65 , 2.45), (2.45, 6)]
results = []

for z_min, z_max in redshift_bins:
    df_bin = df[(df["redshift_qsoc"] >= z_min) & (df["redshift_qsoc"] < z_max)]

    if len(df_bin) == 0:
        continue
      
    posterior = toroidal_dipole_model(df_bin)

    summary = az.summary(
        posterior,
        var_names=["ax", "ay", "az", "Rx", "Ry", "Rz"],
        round_to=5
    )

    # Quick convergence check
    print(summary["r_hat"])

    ax = posterior.posterior["ax"].values
    ay = posterior.posterior["ay"].values
    azv = posterior.posterior["az"].values
    a_total = np.sqrt(ax**2 + ay**2 + azv**2)

    a_total_mean = np.mean(a_total)
    a_total_sd   = np.std(a_total)

    Rx = posterior.posterior["Rx"].values
    Ry = posterior.posterior["Ry"].values
    Rz = posterior.posterior["Rz"].values
    R_total = np.sqrt(Rx**2 + Ry**2 + Rz**2)

    R_total_mean = np.mean(R_total)
    R_total_sd   = np.std(R_total)

    results.append({
        "Redshift Bin": f"{z_min}-{z_max}",

        "ax_mean": summary.loc["ax", "mean"],
        "ay_mean": summary.loc["ay", "mean"],
        "az_mean": summary.loc["az", "mean"],
        "a_total_mean": a_total_mean,

        "ax_sd": summary.loc["ax", "sd"],
        "ay_sd": summary.loc["ay", "sd"],
        "az_sd": summary.loc["az", "sd"],
        "a_total_sd": a_total_sd,

        "Rx_mean": summary.loc["Rx", "mean"],
        "Ry_mean": summary.loc["Ry", "mean"],
        "Rz_mean": summary.loc["Rz", "mean"],
        "R_total_mean": R_total_mean,

        "Rx_sd": summary.loc["Rx", "sd"],
        "Ry_sd": summary.loc["Ry", "sd"],
        "Rz_sd": summary.loc["Rz", "sd"],
        "R_total_sd": R_total_sd,
    })

results_amplitudes = pd.DataFrame(results)
results_amplitudes.to_csv("amplitudes_toroidal_dipole_redshift_3_bins_NUTS_MV.csv", index=False)

#print(results_amplitudes)

