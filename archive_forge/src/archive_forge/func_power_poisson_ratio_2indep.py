import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def power_poisson_ratio_2indep(rate1, rate2, nobs1, nobs_ratio=1, exposure=1, value=0, alpha=0.05, dispersion=1, alternative='smaller', method_var='alt', return_results=True):
    """Power of test of ratio of 2 independent poisson rates.

    This is based on Zhu and Zhu and Lakkis. It does not directly correspond
    to `test_poisson_2indep`.

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
    exposure : float
        Exposure for each observation. Total exposure is nobs1 * exposure
        and nobs2 * exposure.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Rate ratio, rate1 / rate2, under the null hypothesis.
    dispersion : float
        Dispersion coefficient for quasi-Poisson. Dispersion different from
        one can capture over or under dispersion relative to Poisson
        distribution.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates under the alternative can be either equal to the rates under the
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
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    from statsmodels.stats.power import normal_power_het
    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])
    nobs2 = nobs_ratio * nobs1
    v1 = dispersion / exposure * (1 / rate1 + 1 / (nobs_ratio * rate2))
    if method_var == 'alt':
        v0 = v1
    elif method_var == 'score':
        v0 = dispersion / exposure * (1 + value / nobs_ratio) ** 2
        v0 /= value / nobs_ratio * (rate1 + nobs_ratio * rate2)
    else:
        raise NotImplementedError(f'method_var {method_var} not recognized')
    std_null = np.sqrt(v0)
    std_alt = np.sqrt(v1)
    es = np.log(rate1 / rate2) - np.log(value)
    pow_ = normal_power_het(es, nobs1, alpha, std_null=std_null, std_alternative=std_alt, alternative=alternative)
    p_pooled = None
    if return_results:
        res = HolderTuple(power=pow_, p_pooled=p_pooled, std_null=std_null, std_alt=std_alt, nobs1=nobs1, nobs2=nobs2, nobs_ratio=nobs_ratio, alpha=alpha, tuple_=('power',))
        return res
    return pow_