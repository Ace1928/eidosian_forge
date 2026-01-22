import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def confint_poisson(count, exposure, method=None, alpha=0.05):
    """Confidence interval for a Poisson mean or rate

    The function is vectorized for all methods except "midp-c", which uses
    an iterative method to invert the hypothesis test function.

    All current methods are central, that is the probability of each tail is
    smaller or equal to alpha / 2. The one-sided interval limits can be
    obtained by doubling alpha.

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
        This will likely change.
    method : str
        Method to use for confidence interval
        This is required, there is currently no default method
    alpha : float in (0, 1)
        Significance level, nominal coverage of the confidence interval is
        1 - alpha.

    Returns
    -------
    tuple (low, upp) : confidence limits.

    Notes
    -----
    Methods are mainly based on Barker (2002) [1]_ and Swift (2009) [3]_.

    Available methods are:

    - "exact-c" central confidence interval based on gamma distribution
    - "score" : based on score test, uses variance under null value
    - "wald" : based on wald test, uses variance base on estimated rate.
    - "waldccv" : based on wald test with 0.5 count added to variance
      computation. This does not use continuity correction for the center of
      the confidence interval.
    - "midp-c" : based on midp correction of central exact confidence interval.
      this uses numerical inversion of the test function. not vectorized.
    - "jeffreys" : based on Jeffreys' prior. computed using gamma distribution
    - "sqrt" : based on square root transformed counts
    - "sqrt-a" based on Anscombe square root transformation of counts + 3/8.
    - "sqrt-centcc" will likely be dropped. anscombe with continuity corrected
      center.
      (Similar to R survival cipoisson, but without the 3/8 right shift of
      the confidence interval).

    sqrt-cent is the same as sqrt-a, using a different computation, will be
    deleted.

    sqrt-v is a corrected square root method attributed to vandenbrouke, which
    might also be deleted.

    Todo:

    - missing dispersion,
    - maybe split nobs and exposure (? needed in NB). Exposure could be used
      to standardize rate.
    - modified wald, switch method if count=0.

    See Also
    --------
    test_poisson

    References
    ----------
    .. [1] Barker, Lawrence. 2002. “A Comparison of Nine Confidence Intervals
       for a Poisson Parameter When the Expected Number of Events Is ≤ 5.”
       The American Statistician 56 (2): 85–89.
       https://doi.org/10.1198/000313002317572736.
    .. [2] Patil, VV, and HV Kulkarni. 2012. “Comparison of Confidence
       Intervals for the Poisson Mean: Some New Aspects.”
       REVSTAT–Statistical Journal 10(2): 211–27.
    .. [3] Swift, Michael Bruce. 2009. “Comparison of Confidence Intervals for
       a Poisson Mean – Further Considerations.” Communications in Statistics -
       Theory and Methods 38 (5): 748–59.
       https://doi.org/10.1080/03610920802255856.

    """
    n = exposure
    rate = count / exposure
    alpha = alpha / 2
    if method is None:
        msg = 'method needs to be specified, currently no default method'
        raise ValueError(msg)
    if method == 'wald':
        whalf = stats.norm.isf(alpha) * np.sqrt(rate / n)
        ci = (rate - whalf, rate + whalf)
    elif method == 'waldccv':
        whalf = stats.norm.isf(alpha) * np.sqrt((rate + 0.5 / n) / n)
        ci = (rate - whalf, rate + whalf)
    elif method == 'score':
        crit = stats.norm.isf(alpha)
        center = count + crit ** 2 / 2
        whalf = crit * np.sqrt(count + crit ** 2 / 4)
        ci = ((center - whalf) / n, (center + whalf) / n)
    elif method == 'midp-c':
        ci = _invert_test_confint(count, n, alpha=2 * alpha, method='midp-c', method_start='exact-c')
    elif method == 'sqrt':
        crit = stats.norm.isf(alpha)
        center = rate + crit ** 2 / (4 * n)
        whalf = crit * np.sqrt(rate / n)
        ci = (center - whalf, center + whalf)
    elif method == 'sqrt-cent':
        crit = stats.norm.isf(alpha)
        center = count + crit ** 2 / 4
        whalf = crit * np.sqrt(count + 3 / 8)
        ci = ((center - whalf) / n, (center + whalf) / n)
    elif method == 'sqrt-centcc':
        crit = stats.norm.isf(alpha)
        center_low = np.sqrt(np.maximum(count + 3 / 8 - 0.5, 0))
        center_upp = np.sqrt(count + 3 / 8 + 0.5)
        whalf = crit / 2
        ci = ((np.maximum(center_low - whalf, 0) ** 2 - 3 / 8) / n, ((center_upp + whalf) ** 2 - 3 / 8) / n)
    elif method == 'sqrt-a':
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + 3 / 8)
        whalf = crit / 2
        ci = ((np.maximum(center - whalf, 0) ** 2 - 3 / 8) / n, ((center + whalf) ** 2 - 3 / 8) / n)
    elif method == 'sqrt-v':
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + (crit ** 2 + 2) / 12)
        whalf = crit / 2
        ci = (np.maximum(center - whalf, 0) ** 2 / n, (center + whalf) ** 2 / n)
    elif method in ['gamma', 'exact-c']:
        low = stats.gamma.ppf(alpha, count) / exposure
        upp = stats.gamma.isf(alpha, count + 1) / exposure
        if np.isnan(low).any():
            if np.size(low) == 1:
                low = 0.0
            else:
                low[np.isnan(low)] = 0.0
        ci = (low, upp)
    elif method.startswith('jeff'):
        countc = count + 0.5
        ci = (stats.gamma.ppf(alpha, countc) / exposure, stats.gamma.isf(alpha, countc) / exposure)
    else:
        raise ValueError('unknown method %s' % method)
    ci = (np.maximum(ci[0], 0), ci[1])
    return ci