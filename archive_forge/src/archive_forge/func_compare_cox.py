from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def compare_cox(results_x, results_z, store=False):
    """
    Compute the Cox test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    store : bool, default False
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see [1]_ for more information.

    Formulas from [1]_, section 8.3.4 translated to code

    Matches results for Example 8.3 in Greene

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    if _check_nested_results(results_x, results_z):
        raise ValueError(NESTED_ERROR.format(test='Cox comparison'))
    x = results_x.model.exog
    z = results_z.model.exog
    nobs = results_x.model.endog.shape[0]
    sigma2_x = results_x.ssr / nobs
    sigma2_z = results_z.ssr / nobs
    yhat_x = results_x.fittedvalues
    res_dx = OLS(yhat_x, z).fit()
    err_zx = res_dx.resid
    res_xzx = OLS(err_zx, x).fit()
    err_xzx = res_xzx.resid
    sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
    c01 = nobs / 2.0 * (np.log(sigma2_z) - np.log(sigma2_zx))
    v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx ** 2
    q = c01 / np.sqrt(v01)
    pval = 2 * stats.norm.sf(np.abs(q))
    if store:
        res = ResultsStore()
        res.res_dx = res_dx
        res.res_xzx = res_xzx
        res.c01 = c01
        res.v01 = v01
        res.q = q
        res.pvalue = pval
        res.dist = stats.norm
        return (q, pval, res)
    return (q, pval)