import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def conditional_moment_test_generic(mom_test, mom_test_deriv, mom_incl, mom_incl_deriv, var_mom_all=None, cov_type='OPG', cov_kwds=None):
    """generic conditional moment test

    This is mainly intended as internal function in support of diagnostic
    and specification tests. It has no conversion and checking of correct
    arguments.

    Parameters
    ----------
    mom_test : ndarray, 2-D (nobs, k_constraints)
        moment conditions that will be tested to be zero
    mom_test_deriv : ndarray, 2-D, square (k_constraints, k_constraints)
        derivative of moment conditions under test with respect to the
        parameters of the model summed over observations.
    mom_incl : ndarray, 2-D (nobs, k_params)
        moment conditions that where use in estimation, assumed to be zero
        This is score_obs in the case of (Q)MLE
    mom_incl_deriv : ndarray, 2-D, square (k_params, k_params)
        derivative of moment conditions of estimator summed over observations
        This is the information matrix or Hessian in the case of (Q)MLE.
    var_mom_all : None, or ndarray, 2-D, (k, k) with k = k_constraints + k_params
        Expected product or variance of the joint (column_stacked) moment
        conditions. The stacking should have the variance of the moment
        conditions under test in the first k_constraint rows and columns.
        If it is not None, then it will be estimated based on cov_type.
        I think: This is the Hessian of the extended or alternative model
        under full MLE and score test assuming information matrix identity
        holds.

    Returns
    -------
    results

    Notes
    -----
    TODO: cov_type other than OPG is missing
    initial implementation based on Cameron Trived countbook 1998 p.48, p.56

    also included: mom_incl can be None if expected mom_test_deriv is zero.

    References
    ----------
    Cameron and Trivedi 1998 count book
    Wooldridge ???
    Pagan and Vella 1989
    """
    if cov_type != 'OPG':
        raise NotImplementedError
    k_constraints = mom_test.shape[1]
    if mom_incl is None:
        if var_mom_all is None:
            var_cm = mom_test.T.dot(mom_test)
        else:
            var_cm = var_mom_all
    else:
        if var_mom_all is None:
            mom_all = np.column_stack((mom_test, mom_incl))
            var_mom_all = mom_all.T.dot(mom_all)
        tmp = mom_test_deriv.dot(np.linalg.pinv(mom_incl_deriv))
        h = np.column_stack((np.eye(k_constraints), -tmp))
        var_cm = h.dot(var_mom_all.dot(h.T))
    var_cm_inv = np.linalg.pinv(var_cm)
    mom_test_sum = mom_test.sum(0)
    statistic = mom_test_sum.dot(var_cm_inv.dot(mom_test_sum))
    pval = stats.chi2.sf(statistic, k_constraints)
    se = np.sqrt(np.diag(var_cm))
    tvalues = mom_test_sum / se
    pvalues = stats.norm.sf(np.abs(tvalues))
    res = ResultsGeneric(var_cm=var_cm, stat_cmt=statistic, pval_cmt=pval, tvalues=tvalues, pvalues=pvalues)
    return res