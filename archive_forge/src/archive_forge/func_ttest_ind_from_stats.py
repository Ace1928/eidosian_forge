import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True, alternative='two-sided'):
    """
    T-test for means of two independent samples from descriptive statistics.

    This is a test for the null hypothesis that two independent
    samples have identical average (expected) values.

    Parameters
    ----------
    mean1 : array_like
        The mean(s) of sample 1.
    std1 : array_like
        The corrected sample standard deviation of sample 1 (i.e. ``ddof=1``).
    nobs1 : array_like
        The number(s) of observations of sample 1.
    mean2 : array_like
        The mean(s) of sample 2.
    std2 : array_like
        The corrected sample standard deviation of sample 2 (i.e. ``ddof=1``).
    nobs2 : array_like
        The number(s) of observations of sample 2.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions are unequal.
        * 'less': the mean of the first distribution is less than the
          mean of the second distribution.
        * 'greater': the mean of the first distribution is greater than the
          mean of the second distribution.

        .. versionadded:: 1.6.0

    Returns
    -------
    statistic : float or array
        The calculated t-statistics.
    pvalue : float or array
        The two-tailed p-value.

    See Also
    --------
    scipy.stats.ttest_ind

    Notes
    -----
    The statistic is calculated as ``(mean1 - mean2)/se``, where ``se`` is the
    standard error. Therefore, the statistic will be positive when `mean1` is
    greater than `mean2` and negative when `mean1` is less than `mean2`.

    This method does not check whether any of the elements of `std1` or `std2`
    are negative. If any elements of the `std1` or `std2` parameters are
    negative in a call to this method, this method will return the same result
    as if it were passed ``numpy.abs(std1)`` and ``numpy.abs(std2)``,
    respectively, instead; no exceptions or warnings will be emitted.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

    .. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test

    Examples
    --------
    Suppose we have the summary data for two samples, as follows (with the
    Sample Variance being the corrected sample variance)::

                         Sample   Sample
                   Size   Mean   Variance
        Sample 1    13    15.0     87.5
        Sample 2    11    12.0     39.0

    Apply the t-test to this data (with the assumption that the population
    variances are equal):

    >>> import numpy as np
    >>> from scipy.stats import ttest_ind_from_stats
    >>> ttest_ind_from_stats(mean1=15.0, std1=np.sqrt(87.5), nobs1=13,
    ...                      mean2=12.0, std2=np.sqrt(39.0), nobs2=11)
    Ttest_indResult(statistic=0.9051358093310269, pvalue=0.3751996797581487)

    For comparison, here is the data from which those summary statistics
    were taken.  With this data, we can compute the same result using
    `scipy.stats.ttest_ind`:

    >>> a = np.array([1, 3, 4, 6, 11, 13, 15, 19, 22, 24, 25, 26, 26])
    >>> b = np.array([2, 4, 6, 9, 11, 13, 14, 15, 18, 19, 21])
    >>> from scipy.stats import ttest_ind
    >>> ttest_ind(a, b)
    Ttest_indResult(statistic=0.905135809331027, pvalue=0.3751996797581486)

    Suppose we instead have binary data and would like to apply a t-test to
    compare the proportion of 1s in two independent groups::

                          Number of    Sample     Sample
                    Size    ones        Mean     Variance
        Sample 1    150      30         0.2        0.161073
        Sample 2    200      45         0.225      0.175251

    The sample mean :math:`\\hat{p}` is the proportion of ones in the sample
    and the variance for a binary observation is estimated by
    :math:`\\hat{p}(1-\\hat{p})`.

    >>> ttest_ind_from_stats(mean1=0.2, std1=np.sqrt(0.161073), nobs1=150,
    ...                      mean2=0.225, std2=np.sqrt(0.175251), nobs2=200)
    Ttest_indResult(statistic=-0.5627187905196761, pvalue=0.5739887114209541)

    For comparison, we could compute the t statistic and p-value using
    arrays of 0s and 1s and `scipy.stat.ttest_ind`, as above.

    >>> group1 = np.array([1]*30 + [0]*(150-30))
    >>> group2 = np.array([1]*45 + [0]*(200-45))
    >>> ttest_ind(group1, group2)
    Ttest_indResult(statistic=-0.5627179589855622, pvalue=0.573989277115258)

    """
    mean1 = np.asarray(mean1)
    std1 = np.asarray(std1)
    mean2 = np.asarray(mean2)
    std2 = np.asarray(std2)
    if equal_var:
        df, denom = _equal_var_ttest_denom(std1 ** 2, nobs1, std2 ** 2, nobs2)
    else:
        df, denom = _unequal_var_ttest_denom(std1 ** 2, nobs1, std2 ** 2, nobs2)
    res = _ttest_ind_from_stats(mean1, mean2, denom, df, alternative)
    return Ttest_indResult(*res)