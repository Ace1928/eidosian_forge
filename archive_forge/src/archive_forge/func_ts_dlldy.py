import numpy as np
from scipy import special
from scipy.special import gammaln
def ts_dlldy(y, df):
    """derivative of log pdf of standard t with respect to y

    Parameters
    ----------
    y : array_like
        data points of random variable at which loglike is evaluated
    df : array_like
        degrees of freedom,shape parameters of log-likelihood function
        of t distribution

    Returns
    -------
    dlldy : ndarray
        derivative of loglikelihood wrt random variable y evaluated at the
        points given in y

    Notes
    -----
    with mean 0 and scale 1, but variance is df/(df-2)

    """
    df = df * 1.0
    return -(df + 1) / df / (1 + y ** 2 / df) * y