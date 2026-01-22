import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_normalitytests():
    assert_raises(ValueError, stats.skewtest, 4.0)
    assert_raises(ValueError, stats.kurtosistest, 4.0)
    assert_raises(ValueError, stats.normaltest, 4.0)
    st_normal, st_skew, st_kurt = (3.92371918, 1.98078826, -0.01403734)
    pv_normal, pv_skew, pv_kurt = (0.14059673, 0.04761502, 0.98880019)
    pv_skew_less, pv_kurt_less = (1 - pv_skew / 2, pv_kurt / 2)
    pv_skew_greater, pv_kurt_greater = (pv_skew / 2, 1 - pv_kurt / 2)
    x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
    attributes = ('statistic', 'pvalue')
    assert_array_almost_equal(stats.normaltest(x), (st_normal, pv_normal))
    check_named_results(stats.normaltest(x), attributes)
    assert_array_almost_equal(stats.skewtest(x), (st_skew, pv_skew))
    assert_array_almost_equal(stats.skewtest(x, alternative='less'), (st_skew, pv_skew_less))
    assert_array_almost_equal(stats.skewtest(x, alternative='greater'), (st_skew, pv_skew_greater))
    check_named_results(stats.skewtest(x), attributes)
    assert_array_almost_equal(stats.kurtosistest(x), (st_kurt, pv_kurt))
    assert_array_almost_equal(stats.kurtosistest(x, alternative='less'), (st_kurt, pv_kurt_less))
    assert_array_almost_equal(stats.kurtosistest(x, alternative='greater'), (st_kurt, pv_kurt_greater))
    check_named_results(stats.kurtosistest(x), attributes)
    a1 = stats.skewnorm.rvs(a=1, size=10000, random_state=123)
    pval = stats.skewtest(a1, alternative='greater').pvalue
    assert_almost_equal(pval, 0.0, decimal=5)
    a2 = stats.laplace.rvs(size=10000, random_state=123)
    pval = stats.kurtosistest(a2, alternative='greater').pvalue
    assert_almost_equal(pval, 0.0)
    assert_array_almost_equal(stats.normaltest(x, axis=None), (st_normal, pv_normal))
    assert_array_almost_equal(stats.skewtest(x, axis=None), (st_skew, pv_skew))
    assert_array_almost_equal(stats.kurtosistest(x, axis=None), (st_kurt, pv_kurt))
    x = np.arange(10.0)
    x[9] = np.nan
    with np.errstate(invalid='ignore'):
        assert_array_equal(stats.skewtest(x), (np.nan, np.nan))
    expected = (1.018464355396213, 0.308457331951535)
    assert_array_almost_equal(stats.skewtest(x, nan_policy='omit'), expected)
    a1[10:100] = np.nan
    z, p = stats.skewtest(a1, nan_policy='omit')
    zl, pl = stats.skewtest(a1, nan_policy='omit', alternative='less')
    zg, pg = stats.skewtest(a1, nan_policy='omit', alternative='greater')
    assert_allclose(zl, z, atol=1e-15)
    assert_allclose(zg, z, atol=1e-15)
    assert_allclose(pl, 1 - p / 2, atol=1e-15)
    assert_allclose(pg, p / 2, atol=1e-15)
    with np.errstate(all='ignore'):
        assert_raises(ValueError, stats.skewtest, x, nan_policy='raise')
    assert_raises(ValueError, stats.skewtest, x, nan_policy='foobar')
    assert_raises(ValueError, stats.skewtest, list(range(8)), alternative='foobar')
    x = np.arange(30.0)
    x[29] = np.nan
    with np.errstate(all='ignore'):
        assert_array_equal(stats.kurtosistest(x), (np.nan, np.nan))
    expected = (-2.2683547379505273, 0.023307594135872967)
    assert_array_almost_equal(stats.kurtosistest(x, nan_policy='omit'), expected)
    a2[10:20] = np.nan
    z, p = stats.kurtosistest(a2[:100], nan_policy='omit')
    zl, pl = stats.kurtosistest(a2[:100], nan_policy='omit', alternative='less')
    zg, pg = stats.kurtosistest(a2[:100], nan_policy='omit', alternative='greater')
    assert_allclose(zl, z, atol=1e-15)
    assert_allclose(zg, z, atol=1e-15)
    assert_allclose(pl, 1 - p / 2, atol=1e-15)
    assert_allclose(pg, p / 2, atol=1e-15)
    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='raise')
    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='foobar')
    assert_raises(ValueError, stats.kurtosistest, list(range(20)), alternative='foobar')
    with np.errstate(all='ignore'):
        assert_array_equal(stats.normaltest(x), (np.nan, np.nan))
    expected = (6.226040951428745, 0.04446644248650191)
    assert_array_almost_equal(stats.normaltest(x, nan_policy='omit'), expected)
    assert_raises(ValueError, stats.normaltest, x, nan_policy='raise')
    assert_raises(ValueError, stats.normaltest, x, nan_policy='foobar')
    counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
    x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
    assert_equal(stats.kurtosistest(x)[1] < 0.01, True)