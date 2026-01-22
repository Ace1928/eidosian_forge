import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def test_poisson_zeroinflation_jh(results_poisson, exog_infl=None):
    """score test for zero inflation or deflation in Poisson

    This implements Jansakul and Hinde 2009 score test
    for excess zeros against a zero modified Poisson
    alternative. They use a linear link function for the
    inflation model to allow for zero deflation.

    Parameters
    ----------
    results_poisson: results instance
        The test is only valid if the results instance is a Poisson
        model.
    exog_infl : ndarray
        Explanatory variables for the zero inflated or zero modified
        alternative. I exog_infl is None, then the inflation
        probability is assumed to be constant.

    Returns
    -------
    score test results based on chisquare distribution

    Notes
    -----
    This is a score test based on the null hypothesis that
    the true model is Poisson. It will also reject for
    other deviations from a Poisson model if those affect
    the zero probabilities, e.g. in the direction of
    excess dispersion as in the Negative Binomial
    or Generalized Poisson model.
    Therefore, rejection in this test does not imply that
    zero-inflated Poisson is the appropriate model.

    Status: experimental, no verified unit tests,

    TODO: If the zero modification probability is assumed
    to be constant under the alternative, then we only have
    a scalar test score and we can use one-sided tests to
    distinguish zero inflation and deflation from the
    two-sided deviations. (The general one-sided case is
    difficult.)
    In this case the test specializes to the test by Broek

    References
    ----------
    .. [1] Jansakul, N., and J. P. Hinde. 2002. “Score Tests for Zero-Inflated
           Poisson Models.” Computational Statistics & Data Analysis 40 (1):
           75–96. https://doi.org/10.1016/S0167-9473(01)00104-9.
    """
    if not isinstance(results_poisson.model, Poisson):
        import warnings
        warnings.warn('Test is only valid if model is Poisson')
    nobs = results_poisson.model.endog.shape[0]
    if exog_infl is None:
        exog_infl = np.ones((nobs, 1))
    endog = results_poisson.model.endog
    exog = results_poisson.model.exog
    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)
    cov_poi = results_poisson.cov_params()
    cross_derivative = (exog_infl.T * -mu).dot(exog).T
    cov_infl = (exog_infl.T * ((1 - prob_zero) / prob_zero)).dot(exog_infl)
    score_obs_infl = exog_infl * (((endog == 0) - prob_zero) / prob_zero)[:, None]
    score_infl = score_obs_infl.sum(0)
    cov_score_infl = cov_infl - cross_derivative.T.dot(cov_poi).dot(cross_derivative)
    cov_score_infl_inv = np.linalg.pinv(cov_score_infl)
    statistic = score_infl.dot(cov_score_infl_inv).dot(score_infl)
    df2 = np.linalg.matrix_rank(cov_score_infl)
    df = exog_infl.shape[1]
    pvalue = stats.chi2.sf(statistic, df)
    res = HolderTuple(statistic=statistic, pvalue=pvalue, df=df, rank_score=df2, distribution='chi2')
    return res