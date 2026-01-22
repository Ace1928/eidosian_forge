import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def power_poisson_diff_2indep(rate1, rate2, nobs1, nobs_ratio=1, alpha=0.05, value=0, method_var='score', alternative='two-sided', return_results=True):
    """Power of ztest for the difference between two independent poisson rates.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Difference between rates 1 and 2 under the null hypothesis.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates uder the alternative, can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))

    References
    ----------
    .. [1] Stucke, Kathrin, and Meinhard Kieser. 2013. “Sample Size
       Calculations for Noninferiority Trials with Poisson Distributed Count
       Data.” Biometrical Journal 55 (2): 203–16.
       https://doi.org/10.1002/bimj.201200142.
    .. [2] PASS manual chapter 436

    """
    from statsmodels.stats.power import normal_power_het
    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])
    diff = rate1 - rate2
    _, std_null, std_alt = _std_2poisson_power(rate1, rate2, nobs_ratio=nobs_ratio, alpha=alpha, value=value, method_var=method_var)
    pow_ = normal_power_het(diff - value, nobs1, alpha, std_null=std_null, std_alternative=std_alt, alternative=alternative)
    if return_results:
        res = HolderTuple(power=pow_, rates_alt=(rate2 + diff, rate2), std_null=std_null, std_alt=std_alt, nobs1=nobs1, nobs2=nobs_ratio * nobs1, nobs_ratio=nobs_ratio, alpha=alpha, tuple_=('power',))
        return res
    else:
        return pow_