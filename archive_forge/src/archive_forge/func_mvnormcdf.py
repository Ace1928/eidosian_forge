import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
def mvnormcdf(upper, mu, cov, lower=None, **kwds):
    """multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats._mvn.mvndst which calculates
    a rectangular integral over a multivariate normal distribution.

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    mu : array_lik, 1d
       list or array of means
    cov : array_like, 2d
       specifies covariance matrix
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    This function normalizes the location and scale of the multivariate
    normal distribution and then uses `mvstdnormcdf` to call the integration.

    See Also
    --------
    mvstdnormcdf : location and scale standardized multivariate normal cdf
    """
    upper = np.array(upper)
    if lower is None:
        lower = -np.ones(upper.shape) * np.inf
    else:
        lower = np.array(lower)
    cov = np.array(cov)
    stdev = np.sqrt(np.diag(cov))
    lower = (lower - mu) / stdev
    upper = (upper - mu) / stdev
    divrow = np.atleast_2d(stdev)
    corr = cov / divrow / divrow.T
    return mvstdnormcdf(lower, upper, corr, **kwds)