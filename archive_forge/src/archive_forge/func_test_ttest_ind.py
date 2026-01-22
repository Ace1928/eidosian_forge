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
def test_ttest_ind():
    tr = 1.0912746897927283
    pr = 0.2764781861635188
    tpr = ([tr, -tr], [pr, pr])
    rvs2 = np.linspace(1, 100, 100)
    rvs1 = np.linspace(5, 105, 100)
    rvs1_2D = np.array([rvs1, rvs2])
    rvs2_2D = np.array([rvs2, rvs1])
    t, p = stats.ttest_ind(rvs1, rvs2, axis=0)
    assert_array_almost_equal([t, p], (tr, pr))
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(rvs1, rvs2)), [t, p])
    t, p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0)
    assert_array_almost_equal([t, p], tpr)
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args), [t, p])
    t, p = stats.ttest_ind(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal([t, p], tpr)
    args = _desc_stats(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args), [t, p])
    with suppress_warnings() as sup, np.errstate(invalid='ignore'):
        sup.filter(RuntimeWarning, 'Degrees of freedom <= 0 for slice')
        t, p = stats.ttest_ind(4.0, 3.0)
    assert_(np.isnan(t))
    assert_(np.isnan(p))
    rvs1_3D = np.dstack([rvs1_2D, rvs1_2D, rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D, rvs2_2D, rvs2_2D])
    t, p = stats.ttest_ind(rvs1_3D, rvs2_3D, axis=1)
    assert_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))
    t, p = stats.ttest_ind(np.moveaxis(rvs1_3D, 2, 0), np.moveaxis(rvs2_3D, 2, 0), axis=2)
    assert_array_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))
    assert_raises(ValueError, stats.ttest_ind, rvs1, rvs2, alternative='error')
    assert_raises(ValueError, stats.ttest_ind_from_stats, *_desc_stats(rvs1_2D.T, rvs2_2D.T), alternative='error')
    t, p = stats.ttest_ind(rvs1, rvs2, alternative='less')
    assert_allclose(p, 1 - pr / 2)
    assert_allclose(t, tr)
    t, p = stats.ttest_ind(rvs1, rvs2, alternative='greater')
    assert_allclose(p, pr / 2)
    assert_allclose(t, tr)
    t, p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative='less')
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_allclose(stats.ttest_ind_from_stats(*args, alternative='less'), [t, p])
    t, p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative='greater')
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_allclose(stats.ttest_ind_from_stats(*args, alternative='greater'), [t, p])
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    with np.errstate(invalid='ignore'):
        assert_array_equal(stats.ttest_ind(x, y), (np.nan, np.nan))
    assert_array_almost_equal(stats.ttest_ind(x, y, nan_policy='omit'), (0.24779670949091914, 0.8043426733751791))
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='foobar')
    with pytest.warns(RuntimeWarning, match='Precision loss occurred'):
        t, p = stats.ttest_ind([0, 0, 0], [1, 1, 1])
    assert_equal((np.abs(t), p), (np.inf, 0))
    with np.errstate(invalid='ignore'):
        assert_equal(stats.ttest_ind([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_ind(anan, np.zeros((2, 2))), ([0, np.nan], [1, np.nan]))
    rvs1_3D[:, :, 10:15] = np.nan
    rvs2_3D[:, :, 6:12] = np.nan

    def convert(t, p, alt):
        if t < 0 and alt == 'less' or (t > 0 and alt == 'greater'):
            return p / 2
        return 1 - p / 2
    converter = np.vectorize(convert)
    tr, pr = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit')
    t, p = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit', alternative='less')
    assert_allclose(t, tr, rtol=1e-14)
    assert_allclose(p, converter(tr, pr, 'less'), rtol=1e-14)
    t, p = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit', alternative='greater')
    assert_allclose(t, tr, rtol=1e-14)
    assert_allclose(p, converter(tr, pr, 'greater'), rtol=1e-14)