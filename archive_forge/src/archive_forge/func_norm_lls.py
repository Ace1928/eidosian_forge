import numpy as np
from scipy import special
from scipy.special import gammaln
def norm_lls(y, params):
    """normal loglikelihood given observations and mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows

    Returns
    -------
    lls : ndarray
        contribution to loglikelihood for each observation
    """
    mu, sigma2 = params.T
    lls = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (y - mu) ** 2 / sigma2)
    return lls