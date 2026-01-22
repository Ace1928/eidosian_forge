import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def ttest_ind(a, b, axis=0, equal_var=True, alternative='two-sided'):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    equal_var : bool, optional
        If True, perform a standard independent 2 sample test that assumes equal
        population variances.
        If False, perform Welch's t-test, which does not assume equal population
        variance.

        .. versionadded:: 0.17.0
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The p-value.

    Notes
    -----
    For more details on `ttest_ind`, see `scipy.stats.ttest_ind`.

    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if a.size == 0 or b.size == 0:
        return Ttest_indResult(np.nan, np.nan)
    x1, x2 = (a.mean(axis), b.mean(axis))
    v1, v2 = (a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1))
    n1, n2 = (a.count(axis), b.count(axis))
    if equal_var:
        df = ma.asanyarray(n1 + n2 - 2.0)
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        denom = ma.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        df = np.where(np.isnan(df), 1, df)
        denom = ma.sqrt(vn1 + vn2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x1 - x2) / denom
    t, prob = scipy.stats._stats_py._ttest_finish(df, t, alternative)
    return Ttest_indResult(t, prob)