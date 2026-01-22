import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def nonequivalence_poisson_2indep(count1, exposure1, count2, exposure2, low, upp, method='score', compare='ratio'):
    """Test for non-equivalence, minimum effect for poisson.

    This reverses null and alternative hypothesis compared to equivalence
    testing. The null hypothesis is that the effect, ratio (or diff), is in
    an interval that specifies a range of irrelevant or unimportant
    differences between the two samples.

    The Null and alternative hypothesis comparing the ratio of rates are

    for compare = 'ratio':

    - H0: low < rate1 / rate2 < upp
    - H1: rate1 / rate2 <= low or upp <= rate1 / rate2

    for compare = 'diff':

    - H0: rate1 - rate2 <= low or upp <= rate1 - rate2
    - H1: low < rate - rate < upp


    Notes
    -----
    This is implemented as two one-sided tests at the minimum effect boundaries
    (low, upp) with (nominal) size alpha / 2 each.
    The size of the test is the sum of the two one-tailed tests, which
    corresponds to an equal-tailed two-sided test.
    If low and upp are equal, then the result is the same as the standard
    two-sided test.

    The p-value is computed as `2 * min(pvalue_low, pvalue_upp)` in analogy to
    two-sided equal-tail tests.

    In large samples the nominal size of the test will be below alpha.

    References
    ----------
    .. [1] Hodges, J. L., Jr., and E. L. Lehmann. 1954. Testing the Approximate
       Validity of Statistical Hypotheses. Journal of the Royal Statistical
       Society, Series B (Methodological) 16: 261–68.

    .. [2] Kim, Jae H., and Andrew P. Robinson. 2019. “Interval-Based
       Hypothesis Testing and Its Applications to Economics and Finance.”
       Econometrics 7 (2): 21. https://doi.org/10.3390/econometrics7020021.

    """
    tt1 = test_poisson_2indep(count1, exposure1, count2, exposure2, value=low, method=method, compare=compare, alternative='smaller')
    tt2 = test_poisson_2indep(count1, exposure1, count2, exposure2, value=upp, method=method, compare=compare, alternative='larger')
    idx_min = np.asarray(tt1.pvalue < tt2.pvalue, int)
    pvalue = 2 * np.minimum(tt1.pvalue, tt2.pvalue)
    statistic = np.choose(idx_min, [tt1.statistic, tt2.statistic])
    res = HolderTuple(statistic=statistic, pvalue=pvalue, method=method, results_larger=tt1, results_smaller=tt2, title='Equivalence test for 2 independent Poisson rates')
    return res