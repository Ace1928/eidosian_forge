import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def lm_robust_subset(score, k_constraints, score_deriv_inv, cov_score):
    """general formula for score/LM test

    generalized score or lagrange multiplier test for constraints on a subset
    of parameters

    `params_1 = value`, where params_1 is a subset of the unconstrained
    parameter vector.

    It is assumed that all arrays are evaluated at the constrained estimates.

    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    k_constraint : int
        number of constraints
    score_deriv_inv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be OPG or any other estimator if information matrix
        equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    not cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic
    p-value : float
        p-value of the LM test based on chisquare distribution

    Notes
    -----
    The implementation is based on Boos 1992 section 4.1. The same derivation
    is also in other articles and in text books.

    """
    h_uu = score_deriv_inv[:-k_constraints, :-k_constraints]
    h_cu = score_deriv_inv[-k_constraints:, :-k_constraints]
    tmp_proj = h_cu.dot(np.linalg.inv(h_uu))
    tmp = np.column_stack((-tmp_proj, np.eye(k_constraints)))
    cov_score_constraints = tmp.dot(cov_score.dot(tmp.T))
    lm_stat = score.dot(np.linalg.solve(cov_score_constraints, score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval)