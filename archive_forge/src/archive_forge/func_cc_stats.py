import numpy as np
from statsmodels.tools.tools import Bunch
def cc_stats(x1, x2, demean=True):
    """MANOVA statistics based on canonical correlation coefficient

    Calculates Pillai's Trace, Wilk's Lambda, Hotelling's Trace and
    Roy's Largest Root.

    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable.

    Returns
    -------
    res : dict
        Dictionary containing the test statistics.

    Notes
    -----

    same as `canon` in Stata

    missing: F-statistics and p-values

    TODO: should return a results class instead
    produces nans sometimes, singular, perfect correlation of x1, x2 ?

    """
    nobs1, k1 = x1.shape
    nobs2, k2 = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc ** 2
    lam = cc2 / (1 - cc2)
    df_model = k1 * k2
    df_resid = k1 * (nobs1 - k2 - demean)
    s = min(df_model, k1)
    m = 0.5 * (df_model - k1)
    n = 0.5 * (df_resid - k1 - 1)
    df1 = k1 * df_model
    df2 = k2
    pt_value = cc2.sum()
    wl_value = np.product(1 / (1 + lam))
    ht_value = lam.sum()
    rm_value = lam.max()
    res = {}
    res['canonical correlation coefficient'] = cc
    res['eigenvalues'] = lam
    res["Pillai's Trace"] = pt_value
    res["Wilk's Lambda"] = wl_value
    res["Hotelling's Trace"] = ht_value
    res["Roy's Largest Root"] = rm_value
    res['df_resid'] = df_resid
    res['df_m'] = m
    return res