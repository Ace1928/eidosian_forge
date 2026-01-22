import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def confint_quantile_poisson(count, exposure, prob, exposure_new=1.0, method=None, alpha=0.05, alternative='two-sided'):
    """confidence interval for quantile of poisson random variable

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
    prob : float in (0, 1)
        Probability for the quantile, e.g. 0.95 to get the upper 95% quantile.
        With known mean mu, the quantile would be poisson.ppf(prob, mu).
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
    The confidence interval is a closed interval, that is both ``low`` and
    ``upp`` are in the interval.

    See Also
    --------
    confint_poisson
    tolerance_int_poisson

    References
    ----------
    Hahn, Gerald J, and William Q Meeker. 2010. Statistical Intervals: A Guide
    for Practitioners.
    """
    alpha_ = alpha
    if alternative != 'two-sided':
        alpha_ = alpha * 2
    low, upp = confint_poisson(count, exposure, method=method, alpha=alpha_)
    if exposure_new != 1:
        low *= exposure_new
        upp *= exposure_new
    if alternative == 'two-sided':
        low_pred = stats.poisson.ppf(prob, low)
        upp_pred = stats.poisson.ppf(prob, upp)
    elif alternative == 'larger':
        low_pred = 0
        upp_pred = stats.poisson.ppf(prob, upp)
    elif alternative == 'smaller':
        low_pred = stats.poisson.ppf(prob, low)
        upp_pred = np.inf
    low_pred = np.maximum(low_pred, 0)
    return (low_pred, upp_pred)