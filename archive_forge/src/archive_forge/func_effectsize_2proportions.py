import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def effectsize_2proportions(count1, nobs1, count2, nobs2, statistic='diff', zero_correction=None, zero_kwds=None):
    """Effects sizes for two sample binomial proportions

    Parameters
    ----------
    count1, nobs1, count2, nobs2 : array_like
        data for two samples
    statistic : {"diff", "odds-ratio", "risk-ratio", "arcsine"}
        statistic for the comparison of two proportions
        Effect sizes for "odds-ratio" and "risk-ratio" are in logarithm.
    zero_correction : {None, float, "tac", "clip"}
        Some statistics are not finite when zero counts are in the data.
        The options to remove zeros are:

        * float : if zero_correction is a single float, then it will be added
          to all count (cells) if the sample has any zeros.
        * "tac" : treatment arm continuity correction see Ruecker et al 2009,
          section 3.2
        * "clip" : clip proportions without adding a value to all cells
          The clip bounds can be set with zero_kwds["clip_bounds"]

    zero_kwds : dict
        additional options to handle zero counts
        "clip_bounds" tuple, default (1e-6, 1 - 1e-6) if zero_correction="clip"
        other options not yet implemented

    Returns
    -------
    effect size : array
        Effect size for each sample.
    var_es : array
        Estimate of variance of the effect size

    Notes
    -----
    Status: API is experimental, Options for zero handling is incomplete.

    The names for ``statistics`` keyword can be shortened to "rd", "rr", "or"
    and "as".

    The statistics are defined as:

     - risk difference = p1 - p2
     - log risk ratio = log(p1 / p2)
     - log odds_ratio = log(p1 / (1 - p1) * (1 - p2) / p2)
     - arcsine-sqrt = arcsin(sqrt(p1)) - arcsin(sqrt(p2))

    where p1 and p2 are the estimated proportions in sample 1 (treatment) and
    sample 2 (control).

    log-odds-ratio and log-risk-ratio can be transformed back to ``or`` and
    `rr` using `exp` function.

    See Also
    --------
    statsmodels.stats.contingency_tables
    """
    if zero_correction is None:
        cc1 = cc2 = 0
    elif zero_correction == 'tac':
        nobs_t = nobs1 + nobs2
        cc1 = nobs2 / nobs_t
        cc2 = nobs1 / nobs_t
    elif zero_correction == 'clip':
        clip_bounds = zero_kwds.get('clip_bounds', (1e-06, 1 - 1e-06))
        cc1 = cc2 = 0
    elif zero_correction:
        cc1 = cc2 = zero_correction
    else:
        msg = 'zero_correction not recognized or supported'
        raise NotImplementedError(msg)
    zero_mask1 = (count1 == 0) | (count1 == nobs1)
    zero_mask2 = (count2 == 0) | (count2 == nobs2)
    zmask = np.logical_or(zero_mask1, zero_mask2)
    n1 = nobs1 + (cc1 + cc2) * zmask
    n2 = nobs2 + (cc1 + cc2) * zmask
    p1 = (count1 + cc1) / n1
    p2 = (count2 + cc2) / n2
    if zero_correction == 'clip':
        p1 = np.clip(p1, *clip_bounds)
        p2 = np.clip(p2, *clip_bounds)
    if statistic in ['diff', 'rd']:
        rd = p1 - p2
        rd_var = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        eff = rd
        var_eff = rd_var
    elif statistic in ['risk-ratio', 'rr']:
        log_rr = np.log(p1) - np.log(p2)
        log_rr_var = (1 - p1) / p1 / n1 + (1 - p2) / p2 / n2
        eff = log_rr
        var_eff = log_rr_var
    elif statistic in ['odds-ratio', 'or']:
        log_or = np.log(p1) - np.log(1 - p1) - np.log(p2) + np.log(1 - p2)
        log_or_var = 1 / (p1 * (1 - p1) * n1) + 1 / (p2 * (1 - p2) * n2)
        eff = log_or
        var_eff = log_or_var
    elif statistic in ['arcsine', 'arcsin', 'as']:
        as_ = np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))
        as_var = (1 / n1 + 1 / n2) / 4
        eff = as_
        var_eff = as_var
    else:
        msg = 'statistic not recognized, use one of "rd", "rr", "or", "as"'
        raise NotImplementedError(msg)
    return (eff, var_eff)