import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def conditional_moment_test_regression(mom_test, mom_test_deriv=None, mom_incl=None, mom_incl_deriv=None, var_mom_all=None, demean=False, cov_type='OPG', cov_kwds=None):
    """generic conditional moment test based artificial regression

    this is very experimental, no options implemented yet

    so far
    OPG regression, or
    artificial regression with Robust Wald test

    The latter is (as far as I can see) the same as an overidentifying test
    in GMM where the test statistic is the value of the GMM objective function
    and it is assumed that parameters were estimated with optimial GMM, i.e.
    the weight matrix equal to the expectation of the score variance.
    """
    nobs, k_constraints = mom_test.shape
    endog = np.ones(nobs)
    if mom_incl is not None:
        ex = np.column_stack((mom_test, mom_incl))
    else:
        ex = mom_test
    if demean:
        ex -= ex.mean(0)
    if cov_type == 'OPG':
        res = OLS(endog, ex).fit()
        statistic = nobs * res.rsquared
        pval = stats.chi2.sf(statistic, k_constraints)
    else:
        res = OLS(endog, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        tres = res.wald_test(np.eye(ex.shape[1]))
        statistic = tres.statistic
        pval = tres.pvalue
    return (statistic, pval)