import numpy as np
from scipy import special
from scipy.special import gammaln
def normgrad(y, x, params):
    """Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable with mean x*beta, and variance sigma2
    x : ndarray, 2d
        explanatory variables, observation in rows, variables in columns
    params : array_like, (nvars + 1)
        array of coefficients and variance (beta, sigma2)

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt scale (sigma) in second column
    assume params = (beta, sigma2)

    Notes
    -----
    TODO: for heteroscedasticity need sigma to be a 1d array

    """
    beta = params[:-1]
    sigma2 = params[-1] * np.ones((len(y), 1))
    dmudbeta = mean_grad(x, beta)
    mu = np.dot(x, beta)
    params2 = np.column_stack((mu, sigma2))
    dllsdms = norm_lls_grad(y, params2)
    grad = np.column_stack((dllsdms[:, :1] * dmudbeta, dllsdms[:, :1]))
    return grad