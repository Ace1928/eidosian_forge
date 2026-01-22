import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def lm_test_glm(result, exog_extra, mean_deriv=None):
    """score/lagrange multiplier test for GLM

    Wooldridge procedure for test of mean function in GLM

    Parameters
    ----------
    results : GLMResults instance
        results instance with the constrained model
    exog_extra : ndarray or None
        additional exogenous variables for variable addition test
        This can be set to None if mean_deriv is provided.
    mean_deriv : None or ndarray
        Extra moment condition that correspond to the partial derivative of
        a mean function with respect to some parameters.

    Returns
    -------
    test_results : Results instance
        The results instance has the following attributes which are score
        statistic and p-value for 3 versions of the score test.

        c1, pval1 : nonrobust score_test results
        c2, pval2 : score test results robust to over or under dispersion
        c3, pval3 : score test results fully robust to any heteroscedasticity

        The test results instance also has a simple summary method.

    Notes
    -----
    TODO: add `df` to results and make df detection more robust

    This implements the auxiliary regression procedure of Wooldridge,
    implemented based on the presentation in chapter 8 in Handbook of
    Applied Econometrics 2.

    References
    ----------
    Wooldridge, Jeffrey M. 1997. “Quasi-Likelihood Methods for Count Data.”
    Handbook of Applied Econometrics 2: 352–406.

    and other articles and text book by Wooldridge

    """
    if hasattr(result, '_result'):
        res = result._result
    else:
        res = result
    mod = result.model
    nobs = mod.endog.shape[0]
    dlinkinv = mod.family.link.inverse_deriv
    dm = lambda x, linpred: dlinkinv(linpred)[:, None] * x
    var_func = mod.family.variance
    x = result.model.exog
    x2 = exog_extra
    try:
        lin_pred = res.predict(which='linear')
    except TypeError:
        lin_pred = res.predict(linear=True)
    dm_incl = dm(x, lin_pred)
    if x2 is not None:
        dm_excl = dm(x2, lin_pred)
        if mean_deriv is not None:
            dm_excl = np.column_stack((dm_excl, mean_deriv))
    elif mean_deriv is not None:
        dm_excl = mean_deriv
    else:
        raise ValueError('either exog_extra or mean_deriv have to be provided')
    k_constraint = dm_excl.shape[1]
    fittedvalues = res.predict()
    v = var_func(fittedvalues)
    std = np.sqrt(v)
    res_ols1 = OLS(res.resid_response / std, np.column_stack((dm_incl, dm_excl)) / std[:, None]).fit()
    c1 = res_ols1.ess
    pval1 = stats.chi2.sf(c1, k_constraint)
    c2 = nobs * res_ols1.rsquared
    pval2 = stats.chi2.sf(c2, k_constraint)
    from statsmodels.stats.multivariate_tools import partial_project
    pp = partial_project(dm_excl / std[:, None], dm_incl / std[:, None])
    resid_p = res.resid_response / std
    res_ols3 = OLS(np.ones(nobs), pp.resid * resid_p[:, None]).fit()
    c3b = res_ols3.ess
    pval3 = stats.chi2.sf(c3b, k_constraint)
    tres = TestResults(c1=c1, pval1=pval1, c2=c2, pval2=pval2, c3=c3b, pval3=pval3)
    return tres