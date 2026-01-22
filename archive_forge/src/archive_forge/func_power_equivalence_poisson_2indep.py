import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def power_equivalence_poisson_2indep(rate1, rate2, nobs1, low, upp, nobs_ratio=1, exposure=1, alpha=0.05, dispersion=1, method_var='alt', return_results=False):
    """Power of equivalence test of ratio of 2 independent poisson rates.

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
    low : float
        Lower equivalence margin for the rate ratio, rate1 / rate2.
    upp : float
        Upper equivalence margin for the rate ratio, rate1 / rate2.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    exposure : float
        Exposure for each observation. Total exposure is nobs1 * exposure
        and nobs2 * exposure.
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
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])
    nobs2 = nobs_ratio * nobs1
    v1 = dispersion / exposure * (1 / rate1 + 1 / (nobs_ratio * rate2))
    if method_var == 'alt':
        v0_low = v0_upp = v1
    elif method_var == 'score':
        v0_low = dispersion / exposure * (1 + low * nobs_ratio) ** 2
        v0_low /= low * nobs_ratio * (rate1 + nobs_ratio * rate2)
        v0_upp = dispersion / exposure * (1 + upp * nobs_ratio) ** 2
        v0_upp /= upp * nobs_ratio * (rate1 + nobs_ratio * rate2)
    else:
        raise NotImplementedError(f'method_var {method_var} not recognized')
    es_low = np.log(rate1 / rate2) - np.log(low)
    es_upp = np.log(rate1 / rate2) - np.log(upp)
    std_null_low = np.sqrt(v0_low)
    std_null_upp = np.sqrt(v0_upp)
    std_alternative = np.sqrt(v1)
    pow_ = _power_equivalence_het(es_low, es_upp, nobs2, alpha=alpha, std_null_low=std_null_low, std_null_upp=std_null_upp, std_alternative=std_alternative)
    if return_results:
        res = HolderTuple(power=pow_[0], power_margins=pow[1:], std_null_low=std_null_low, std_null_upp=std_null_upp, std_alt=std_alternative, nobs1=nobs1, nobs2=nobs2, nobs_ratio=nobs_ratio, alpha=alpha, tuple_=('power',))
        return res
    else:
        return pow_[0]