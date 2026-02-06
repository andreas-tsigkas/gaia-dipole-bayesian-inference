-- Gaia DR3 quasar selection query used in the analysis
-- Purpose: select QSO-like sources with five-parameter astrometric solution
-- Note: run in the Gaia Archive ADQL interface

SELECT TOP 2000000
    s.source_id,
    v.ra,
    v.dec,
    v.pmra,
    v.pmdec,
    v.pmra_error,
    v.pmdec_error,
    v.pmra_pmdec_corr,
    v.parallax,
    v.phot_g_mean_mag,
    v.ruwe,
    q.redshift_qsoc
FROM gaiadr3.agn_cross_id AS s
INNER JOIN gaiadr3.gaia_source AS v 
    ON s.source_id = v.source_id
INNER JOIN gaiadr3.qso_candidates AS q
    ON s.source_id = q.source_id
WHERE v.astrometric_params_solved = 31
