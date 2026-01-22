import numpy as np
from numpy import poly1d
from scipy.special import beta
def tukeylambda_kurtosis(lam):
    """Kurtosis of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.25, the variance is not defined, so
        np.nan is returned.  For lam = 0.25, np.inf is returned.

    """
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)
    threshold = 0.055
    low_mask = lam < -0.25
    negqrtr_mask = lam == -0.25
    small_mask = np.abs(lam) < threshold
    reg_mask = ~(low_mask | negqrtr_mask | small_mask)
    small = lam[small_mask]
    reg = lam[reg_mask]
    k = np.empty_like(lam)
    k[low_mask] = np.nan
    k[negqrtr_mask] = np.inf
    if small.size > 0:
        k[small_mask] = _tukeylambda_kurt_p(small) / _tukeylambda_kurt_q(small)
    if reg.size > 0:
        numer = 1.0 / (4 * reg + 1) - 4 * beta(3 * reg + 1, reg + 1) + 3 * beta(2 * reg + 1, 2 * reg + 1)
        denom = 2 * (1.0 / (2 * reg + 1) - beta(reg + 1, reg + 1)) ** 2
        k[reg_mask] = numer / denom - 3
    k.shape = shp
    return k