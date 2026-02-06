"""
This file defines the dipole + rotation (glide + rotation) Bayesian model.

The model is intentionally separated from data access.
To run it on real Gaia DR3 data, the user is expected to:

- query Gaia using astroquery.gaia (see sql/gaia_qso_selection.sql)
- load the result into a pandas DataFrame `df`
- pass `df` to `toroidal_dipole_model(df)`

Commented imports below indicate the libraries used in the full pipeline,
but are not required for defining the model itself.
"""

# Data access libraries used elsewhere in the pipeline
# from astroquery.gaia import Gaia
# import pandas as pd
# from astropy.table import Table

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def toroidal_dipole_model(df):
    """
    Dipole + rotation (glide + rotation) model for Gaia DR3 quasar proper motions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr

    Returns
    -------
    posterior : arviz.InferenceData
        Posterior samples from the Bayesian fit.
    """

    ra_rad  = pt.constant(np.deg2rad(df["ra"].values))
    dec_rad = pt.constant(np.deg2rad(df["dec"].values))

    pmra_obs  = pt.constant(df["pmra"].values)
    pmdec_obs = pt.constant(df["pmdec"].values)

    pmra_err  = df["pmra_error"].values
    pmdec_err = df["pmdec_error"].values
    pm_corr   = df["pmra_pmdec_corr"].values

    n = len(df)

    # Per-source covariance and its analytic inverse
    var_ra     = pmra_err**2
    var_dec    = pmdec_err**2
    cov_ra_dec = pm_corr * pmra_err * pmdec_err

    det_cov = var_ra * var_dec - cov_ra_dec**2

    prec_00 = pt.constant(var_dec / det_cov)
    prec_11 = pt.constant(var_ra / det_cov)
    prec_01 = pt.constant(-cov_ra_dec / det_cov)

    # Precompute constants for the likelihood
    sum_logdet_cov = pt.constant(np.sum(np.log(det_cov)))
    normalization  = pt.constant(n * 2 * np.log(2 * np.pi))

    with pm.Model() as model:
        # Priors for the glide (dipole)
        ax = pm.Normal("ax", mu=0, sigma=1e-2)
        ay = pm.Normal("ay", mu=0, sigma=1e-2)
        az = pm.Normal("az", mu=0, sigma=1e-2)

        # Priors for the rotation
        Rx = pm.Normal("Rx", mu=0, sigma=1e-2)
        Ry = pm.Normal("Ry", mu=0, sigma=1e-2)
        Rz = pm.Normal("Rz", mu=0, sigma=1e-2)

        # Dipole model
        pmra_dipole = (-ax * pt.sin(ra_rad) + ay * pt.cos(ra_rad))
        pmdec_dipole = (
            -ax * pt.sin(dec_rad) * pt.cos(ra_rad)
            - ay * pt.sin(dec_rad) * pt.sin(ra_rad)
            + az * pt.cos(dec_rad)
        )

        # Rotation model
        pmra_toroidal = (
            Rx * pt.sin(dec_rad) * pt.cos(ra_rad)
            + Ry * pt.sin(dec_rad) * pt.sin(ra_rad)
            - Rz * pt.cos(dec_rad)
        )
        pmdec_toroidal = (-Rx * pt.sin(ra_rad) + Ry * pt.cos(ra_rad))

        # Full model
        pmra_model  = pmra_dipole + pmra_toroidal
        pmdec_model = pmdec_dipole + pmdec_toroidal

        # Residuals
        delta_pmra  = pmra_obs  - pmra_model
        delta_pmdec = pmdec_obs - pmdec_model

        # Quadratic form
        quad_form = pt.sum(
            prec_00 * delta_pmra**2
            + 2 * prec_01 * delta_pmra * delta_pmdec
            + prec_11 * delta_pmdec**2
        )
        
        # Custom log-likelihood
        log_likelihood = -0.5 * (quad_form + sum_logdet_cov + normalization)
        pm.Potential("custom_log_likelihood", log_likelihood)

        posterior = pm.sample(2000, tune=2000, chains=8, cores=8)

    return posterior
