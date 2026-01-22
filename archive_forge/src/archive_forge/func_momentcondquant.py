import numpy as np
from scipy import stats, optimize, special
def momentcondquant(distfn, params, mom2, quantile=None, shape=None):
    """moment conditions for estimating distribution parameters by matching
    quantiles, defines as many moment conditions as quantiles.

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical quantiles

    Notes
    -----
    This can be used for method of moments or for generalized method of
    moments.

    """
    if len(params) == 2:
        loc, scale = params
    elif len(params) == 3:
        shape, loc, scale = params
    else:
        pass
    pq, xq = quantile
    cdfdiff = distfn.cdf(xq, *params) - pq
    return cdfdiff