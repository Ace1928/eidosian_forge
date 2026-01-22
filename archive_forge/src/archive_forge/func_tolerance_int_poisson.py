import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def tolerance_int_poisson(count, exposure, prob=0.95, exposure_new=1.0, method=None, alpha=0.05, alternative='two-sided'):
    """tolerance interval for a poisson observation

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
    prob : float in (0, 1)
        Probability of poisson interval, often called "content".
        With known parameters, each tail would have at most probability
        ``1 - prob / 2`` in the two-sided interval.
    exposure_new : float
        Exposure of the new or predicted observation.
    method : str
        Method to used for confidence interval of the estimate of the
        poisson rate, used in `confint_poisson`.
        This is required, there is currently no default method.
    alpha : float in (0, 1)
        Significance level for the confidence interval of the estimate of the
        Poisson rate. Nominal coverage of the confidence interval is
        1 - alpha.
    alternative : {"two-sider", "larger", "smaller")
        The tolerance interval can be two-sided or one-sided.
        Alternative "larger" provides the upper bound of the confidence
        interval, larger counts are outside the interval.

    Returns
    -------
    tuple (low, upp) of limits of tolerance interval.
        The tolerance interval is a closed interval, that is both ``low`` and
        ``upp`` are in the interval.

    Notes
    -----
    verified against R package tolerance `poistol.int`

    See Also
    --------
    confint_poisson
    confint_quantile_poisson

    References
    ----------
    .. [1] Hahn, Gerald J., and William Q. Meeker. 1991. Statistical Intervals:
       A Guide for Practitioners. 1st ed. Wiley Series in Probability and
       Statistics. Wiley. https://doi.org/10.1002/9780470316771.
    .. [2] Hahn, Gerald J., and Ramesh Chandra. 1981. “Tolerance Intervals for
       Poisson and Binomial Variables.” Journal of Quality Technology 13 (2):
       100–110. https://doi.org/10.1080/00224065.1981.11980998.

    """
    prob_tail = 1 - prob
    alpha_ = alpha
    if alternative != 'two-sided':
        alpha_ = alpha * 2
    low, upp = confint_poisson(count, exposure, method=method, alpha=alpha_)
    if exposure_new != 1:
        low *= exposure_new
        upp *= exposure_new
    if alternative == 'two-sided':
        low_pred = stats.poisson.ppf(prob_tail / 2, low)
        upp_pred = stats.poisson.ppf(1 - prob_tail / 2, upp)
    elif alternative == 'larger':
        low_pred = 0
        upp_pred = stats.poisson.ppf(1 - prob_tail, upp)
    elif alternative == 'smaller':
        low_pred = stats.poisson.ppf(prob_tail, low)
        upp_pred = np.inf
    low_pred = np.maximum(low_pred, 0)
    return (low_pred, upp_pred)