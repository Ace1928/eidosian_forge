import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def normal_sample_size_one_tail(diff, power, alpha, std_null=1.0, std_alternative=None):
    """explicit sample size computation if only one tail is relevant

    The sample size is based on the power in one tail assuming that the
    alternative is in the tail where the test has power that increases
    with sample size.
    Use alpha/2 to compute the one tail approximation to the two-sided
    test, i.e. consider only one tail of two-sided test.

    Parameters
    ----------
    diff : float
        difference in the estimated means or statistics under the alternative.
    power : float in interval (0,1)
        power of the test, e.g. 0.8, is one minus the probability of a type II
        error. Power is the probability that the test correctly rejects the
        Null Hypothesis if the Alternative Hypothesis is true.
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
        Note: alpha is used for one tail. Use alpha/2 for two-sided
        alternative.
    std_null : float
        standard deviation under the Null hypothesis without division by
        sqrt(nobs)
    std_alternative : float
        standard deviation under the Alternative hypothesis without division
        by sqrt(nobs). Defaults to None. If None, ``std_alternative`` is set
        to the value of ``std_null``.

    Returns
    -------
    nobs : float
        Sample size to achieve (at least) the desired power.
        If the minimum power is satisfied for all positive sample sizes, then
        ``nobs`` will be zero. This will be the case when power <= alpha if
        std_alternative is equal to std_null.

    """
    if std_alternative is None:
        std_alternative = std_null
    crit_power = stats.norm.isf(power)
    crit = stats.norm.isf(alpha)
    n1 = (np.maximum(crit * std_null - crit_power * std_alternative, 0) / diff) ** 2
    return n1