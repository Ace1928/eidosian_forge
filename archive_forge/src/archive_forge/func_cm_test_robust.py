import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def cm_test_robust(resid, resid_deriv, instruments, weights=1):
    """score/lagrange multiplier of Wooldridge

    generic version of Wooldridge procedure for test of conditional moments

    Limitation: This version allows only for one unconditional moment
    restriction, i.e. resid is scalar for each observation.
    Another limitation is that it assumes independent observations, no
    correlation in residuals and weights cannot be replaced by cross-observation
    whitening.

    Parameters
    ----------
    resid : ndarray, (nobs, )
        conditional moment restriction, E(r | x, params) = 0
    resid_deriv : ndarray, (nobs, k_params)
        derivative of conditional moment restriction with respect to parameters
    instruments : ndarray, (nobs, k_instruments)
        indicator variables of Wooldridge, multiplies the conditional momen
        restriction
    weights : ndarray
        This is a weights function as used in WLS. The moment
        restrictions are multiplied by weights. This corresponds to the
        inverse of the variance in a heteroskedastic model.

    Returns
    -------
    test_results : Results instance
        ???  TODO

    Notes
    -----
    This implements the auxiliary regression procedure of Wooldridge,
    implemented based on procedure 2.1 in Wooldridge 1990.

    Wooldridge allows for multivariate conditional moments (`resid`)
    TODO: check dimensions for multivariate case for extension

    References
    ----------
    Wooldridge
    Wooldridge
    and more Wooldridge

    """
    nobs = resid.shape[0]
    from statsmodels.stats.multivariate_tools import partial_project
    w_sqrt = np.sqrt(weights)
    if np.size(weights) > 1:
        w_sqrt = w_sqrt[:, None]
    pp = partial_project(instruments * w_sqrt, resid_deriv * w_sqrt)
    mom_resid = pp.resid
    moms_test = mom_resid * resid[:, None] * w_sqrt
    k_constraint = moms_test.shape[1]
    cov = moms_test.T.dot(moms_test)
    diff = moms_test.sum(0)
    stat = diff.dot(np.linalg.solve(cov, diff))
    stat2 = OLS(np.ones(nobs), moms_test).fit().ess
    pval = stats.chi2.sf(stat, k_constraint)
    return (stat, pval, stat2)