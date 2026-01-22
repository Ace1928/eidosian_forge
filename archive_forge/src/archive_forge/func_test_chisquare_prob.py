import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def test_chisquare_prob(results, probs, bin_edges=None, method=None):
    """
    chisquare test for predicted probabilities using cmt-opg

    Parameters
    ----------
    results : results instance
        Instance of a count regression results
    probs : ndarray
        Array of predicted probabilities with observations
        in rows and event counts in columns
    bin_edges : None or array
        intervals to combine several counts into cells
        see combine_bins

    Returns
    -------
    (api not stable, replace by test-results class)
    statistic : float
        chisquare statistic for tes
    p-value : float
        p-value of test
    df : int
        degrees of freedom for chisquare distribution
    extras : ???
        currently returns a tuple with some intermediate results
        (diff, res_aux)

    Notes
    -----

    Status : experimental, no verified unit tests, needs to be generalized
    currently only OPG version with auxiliary regression is implemented

    Assumes counts are np.arange(probs.shape[1]), i.e. consecutive
    integers starting at zero.

    Auxiliary regression drops the last column of binned probs to avoid
    that probabilities sum to 1.

    References
    ----------
    .. [1] Andrews, Donald W. K. 1988a. “Chi-Square Diagnostic Tests for
           Econometric Models: Theory.” Econometrica 56 (6): 1419–53.
           https://doi.org/10.2307/1913105.

    .. [2] Andrews, Donald W. K. 1988b. “Chi-Square Diagnostic Tests for
           Econometric Models.” Journal of Econometrics 37 (1): 135–56.
           https://doi.org/10.1016/0304-4076(88)90079-6.

    .. [3] Manjón, M., and O. Martínez. 2014. “The Chi-Squared Goodness-of-Fit
           Test for Count-Data Models.” Stata Journal 14 (4): 798–816.
    """
    res = results
    score_obs = results.model.score_obs(results.params)
    d_ind = (res.model.endog[:, None] == np.arange(probs.shape[1])).astype(int)
    if bin_edges is not None:
        d_ind_bins, k_bins = _combine_bins(bin_edges, d_ind)
        probs_bins, k_bins = _combine_bins(bin_edges, probs)
        k_bins = probs_bins.shape[-1]
    else:
        d_ind_bins, k_bins = (d_ind, d_ind.shape[1])
        probs_bins = probs
    diff1 = d_ind_bins - probs_bins
    x_aux = np.column_stack((score_obs, diff1[:, :-1]))
    nobs = x_aux.shape[0]
    res_aux = OLS(np.ones(nobs), x_aux).fit()
    chi2_stat = nobs * (1 - res_aux.ssr / res_aux.uncentered_tss)
    df = res_aux.model.rank - score_obs.shape[1]
    if df < k_bins - 1:
        import warnings
        warnings.warn('auxiliary model is rank deficient')
    statistic = chi2_stat
    pvalue = stats.chi2.sf(chi2_stat, df)
    res = HolderTuple(statistic=statistic, pvalue=pvalue, df=df, diff1=diff1, res_aux=res_aux, distribution='chi2')
    return res