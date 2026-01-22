import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def lm_robust(score, constraint_matrix, score_deriv_inv, cov_score, cov_params=None):
    """general formula for score/LM test

    generalized score or lagrange multiplier test for implicit constraints

    `r(params) = 0`, with gradient `R = d r / d params`

    linear constraints are given by `R params - q = 0`

    It is assumed that all arrays are evaluated at the constrained estimates.

    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    constraint_matrix R : ndarray
        Linear restriction matrix or Jacobian of nonlinear constraints
    hessian_inv, Ainv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be OPG or any other estimator if information matrix
        equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic

    Notes
    -----

    """
    R, Ainv, B, V = (constraint_matrix, score_deriv_inv, cov_score, cov_params)
    tmp = R.dot(Ainv)
    wscore = tmp.dot(score)
    if B is None and V is None:
        lm_stat = score.dot(Ainv.dot(score))
    else:
        if V is None:
            inner = tmp.dot(B).dot(tmp.T)
        else:
            inner = R.dot(V).dot(R.T)
        lm_stat = wscore.dot(np.linalg.solve(inner, wscore))
    return lm_stat