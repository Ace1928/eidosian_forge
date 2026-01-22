import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def tost_poisson_2indep(count1, exposure1, count2, exposure2, low, upp, method='score', compare='ratio'):
    """Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent poisson samples.

    The Null and alternative hypothesis for equivalence testing are

    for compare = 'ratio'

    - H0: rate1 / rate2 <= low or upp <= rate1 / rate2
    - H1: low < rate1 / rate2 < upp

    for compare = 'diff'

    - H0: rate1 - rate2 <= low or upp <= rate1 - rate2
    - H1: low < rate - rate < upp

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in second sample
    exposure2 : float
        Total exposure (time * subjects) in second sample
    low, upp :
        equivalence margin for the ratio or difference of Poisson rates
    method: string
        TOST uses ``test_poisson_2indep`` and has the same methods.

        ratio:

        - 'wald': method W1A, wald test, variance based on observed rates
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'exact-cond': exact conditional test based on binomial distribution
           This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': midpoint-pvalue of exact conditional test
        - 'etest' or 'etest-score: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'etest-score' or 'etest: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    See Also
    --------
    test_poisson_2indep
    confint_poisson_2indep
    """
    tt1 = test_poisson_2indep(count1, exposure1, count2, exposure2, value=low, method=method, compare=compare, alternative='larger')
    tt2 = test_poisson_2indep(count1, exposure1, count2, exposure2, value=upp, method=method, compare=compare, alternative='smaller')
    idx_max = np.asarray(tt1.pvalue < tt2.pvalue, int)
    statistic = np.choose(idx_max, [tt1.statistic, tt2.statistic])
    pvalue = np.choose(idx_max, [tt1.pvalue, tt2.pvalue])
    res = HolderTuple(statistic=statistic, pvalue=pvalue, method=method, compare=compare, equiv_limits=(low, upp), results_larger=tt1, results_smaller=tt2, title='Equivalence test for 2 independent Poisson rates')
    return res