"""
Bayes factor comparison for Gaia DR3 QSO proper motions.

Compare two models using Bayesian evidence (marginal likelihood) from SMC:

1) Dipole only
2) Dipole plus toroidal rotation

Why SMC?
Bayes factors require the marginal likelihood Z. SMC provides an estimate of log Z. Elsewhere in the project, NUTS is used for posterior inference, which is a different task.

Data
The input DataFrame `df` is assumed to already exist.

Required columns in df
ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr

Note
A further comparison with quadrupole extensions was performed in the paper, but is not shown here.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pymc.smc.kernels import MH


def logZ(idata):
    """
    Extract final SMC log marginal likelihood in a version robust way.
    
    Returns
    float
        Mean final log Z over chains.
    """
    da = idata.sample_stats["log_marginal_likelihood"]

    for dim in ("draw", "stage"):
        if dim in da.dims:
            da = da.isel({dim: -1})

    # After slicing to the final step, average over chains
    # and convert to a float
    flat = [float(np.asarray(v).ravel()[-1]) for v in da.values.ravel()]
    return float(np.mean(flat))


def dipole_model_smc(df, draws=3000, chains=16, cores=16):
    ra_rad  = pt.constant(np.deg2rad(df["ra"].values))
    dec_rad = pt.constant(np.deg2rad(df["dec"].values))

    pmra_obs  = pt.constant(df["pmra"].values)
    pmdec_obs = pt.constant(df["pmdec"].values)

    pmra_err  = df["pmra_error"].values
    pmdec_err = df["pmdec_error"].values
    pm_corr   = df["pmra_pmdec_corr"].values

    n = len(df)

    # Per source covariance entries and determinant
    var_ra     = pmra_err**2
    var_dec    = pmdec_err**2
    cov_ra_dec = pm_corr * pmra_err * pmdec_err
    det_cov    = var_ra * var_dec - cov_ra_dec**2

    # Analytic precision matrix entries for each 2x2 block
    prec_00 = pt.constant(var_dec     / det_cov)
    prec_11 = pt.constant(var_ra      / det_cov)
    prec_01 = pt.constant(-cov_ra_dec / det_cov)

    # Gaussian normalization and log det terms, summed over sources
    sum_logdet_cov = pt.constant(np.log(det_cov).sum())
    normalization  = pt.constant(n * 2 * np.log(2 * np.pi))

    with pm.Model() as model:
        # Dipole parameters
        ax = pm.Normal("ax", 0, 1e-2)
        ay = pm.Normal("ay", 0, 1e-2)
        az = pm.Normal("az", 0, 1e-2)

        # Dipole prediction
        pmra_model = -ax * pt.sin(ra_rad) + ay * pt.cos(ra_rad)
        pmdec_model = (
            -ax * pt.sin(dec_rad) * pt.cos(ra_rad)
            - ay * pt.sin(dec_rad) * pt.sin(ra_rad)
            + az * pt.cos(dec_rad)
        )

        # Residuals
        delta_pmra  = pmra_obs  - pmra_model
        delta_pmdec = pmdec_obs - pmdec_model

        # Sum of quadratic forms
        quad_form = pt.sum(
            prec_00 * delta_pmra**2
            + 2 * prec_01 * delta_pmra * delta_pmdec
            + prec_11 * delta_pmdec**2
        )

        # Custom Gaussian log likelihood
        log_likelihood = -0.5 * (quad_form + sum_logdet_cov + normalization)
        pm.Potential("custom_log_likelihood", log_likelihood)

        # SMC sampler, MH kernel
        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            kernel=MH,
            threshold=0.7,
            correlation_threshold=0.007,
            cores=cores,
            return_inferencedata=True,
        )

    return idata


def dipole_toroidal_model_smc(df, draws=3500, chains=16, cores=16):
    ra_rad  = pt.constant(np.deg2rad(df["ra"].values))
    dec_rad = pt.constant(np.deg2rad(df["dec"].values))

    pmra_obs  = pt.constant(df["pmra"].values)
    pmdec_obs = pt.constant(df["pmdec"].values)

    pmra_err  = df["pmra_error"].values
    pmdec_err = df["pmdec_error"].values
    pm_corr   = df["pmra_pmdec_corr"].values

    n = len(df)

    var_ra     = pmra_err**2
    var_dec    = pmdec_err**2
    cov_ra_dec = pm_corr * pmra_err * pmdec_err
    det_cov    = var_ra * var_dec - cov_ra_dec**2

    prec_00 = pt.constant(var_dec     / det_cov)
    prec_11 = pt.constant(var_ra      / det_cov)
    prec_01 = pt.constant(-cov_ra_dec / det_cov)

    sum_logdet_cov = pt.constant(np.log(det_cov).sum())
    normalization  = pt.constant(n * 2 * np.log(2 * np.pi))

    with pm.Model() as model:
        # Dipole parameters
        ax = pm.Normal("ax", 0, 1e-2)
        ay = pm.Normal("ay", 0, 1e-2)
        az = pm.Normal("az", 0, 1e-2)

        # Toroidal rotation parameters
        Rx = pm.Normal("Rx", 0, 1e-2)
        Ry = pm.Normal("Ry", 0, 1e-2)
        Rz = pm.Normal("Rz", 0, 1e-2)

        # Dipole prediction
        pmra_dipole = -ax * pt.sin(ra_rad) + ay * pt.cos(ra_rad)
        pmdec_dipole = (
            -ax * pt.sin(dec_rad) * pt.cos(ra_rad)
            - ay * pt.sin(dec_rad) * pt.sin(ra_rad)
            + az * pt.cos(dec_rad)
        )

        # Toroidal prediction
        pmra_toroidal = (
            Rx * pt.sin(dec_rad) * pt.cos(ra_rad)
            + Ry * pt.sin(dec_rad) * pt.sin(ra_rad)
            - Rz * pt.cos(dec_rad)
        )
        pmdec_toroidal = -Rx * pt.sin(ra_rad) + Ry * pt.cos(ra_rad)

        # Full model
        pmra_model  = pmra_dipole  + pmra_toroidal
        pmdec_model = pmdec_dipole + pmdec_toroidal

        delta_pmra  = pmra_obs  - pmra_model
        delta_pmdec = pmdec_obs - pmdec_model

        quad_form = pt.sum(
            prec_00 * delta_pmra**2
            + 2 * prec_01 * delta_pmra * delta_pmdec
            + prec_11 * delta_pmdec**2
        )

        log_likelihood = -0.5 * (quad_form + sum_logdet_cov + normalization)
        pm.Potential("custom_log_likelihood", log_likelihood)

        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            kernel=MH,
            threshold=0.7,
            correlation_threshold=0.007,
            cores=cores,
            return_inferencedata=True,
        )

    return idata


# df must already exist here
# df = ...

idata_dip = dipole_model_smc(df)
idata_full = dipole_toroidal_model_smc(df)

# Optional: parameter summaries, keep if you want to show estimates
print(az.summary(idata_dip, var_names=["ax", "ay", "az"], hdi_prob=0.683, round_to=5))
print(az.summary(idata_full, var_names=["ax", "ay", "az", "Rx", "Ry", "Rz"], hdi_prob=0.683, round_to=5))

# Model evidence and Bayes factor
logZ_dip  = logZ(idata_dip)
logZ_full = logZ(idata_full)

log_BF = logZ_full - logZ_dip
BF = np.exp(log_BF)

print(f"log BF (Dip+Tor / Dip) = {log_BF:.2f}")
print(f"BF     (Dip+Tor / Dip) = {BF:.3g}")
