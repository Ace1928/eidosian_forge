import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def corr_equi(k_vars, rho):
    """create equicorrelated correlation matrix with rho on off diagonal

    Parameters
    ----------
    k_vars : int
        number of variables, correlation matrix will be (k_vars, k_vars)
    rho : float
        correlation between any two random variables

    Returns
    -------
    corr : ndarray (k_vars, k_vars)
        correlation matrix

    """
    corr = np.empty((k_vars, k_vars))
    corr.fill(rho)
    corr[np.diag_indices_from(corr)] = 1
    return corr