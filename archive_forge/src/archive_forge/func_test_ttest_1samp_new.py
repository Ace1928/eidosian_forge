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
def test_ttest_1samp_new():
    n1, n2, n3 = (10, 15, 20)
    rvn1 = stats.norm.rvs(loc=5, scale=10, size=(n1, n2, n3))
    t1, p1 = stats.ttest_1samp(rvn1[:, :, :], np.ones((n2, n3)), axis=0)
    t2, p2 = stats.ttest_1samp(rvn1[:, :, :], 1, axis=0)
    t3, p3 = stats.ttest_1samp(rvn1[:, 0, 0], 1)
    assert_array_almost_equal(t1, t2, decimal=14)
    assert_almost_equal(t1[0, 0], t3, decimal=14)
    assert_equal(t1.shape, (n2, n3))
    t1, p1 = stats.ttest_1samp(rvn1[:, :, :], np.ones((n1, 1, n3)), axis=1)
    t2, p2 = stats.ttest_1samp(rvn1[:, :, :], 1, axis=1)
    t3, p3 = stats.ttest_1samp(rvn1[0, :, 0], 1)
    assert_array_almost_equal(t1, t2, decimal=14)
    assert_almost_equal(t1[0, 0], t3, decimal=14)
    assert_equal(t1.shape, (n1, n3))
    t1, p1 = stats.ttest_1samp(rvn1[:, :, :], np.ones((n1, n2, 1)), axis=2)
    t2, p2 = stats.ttest_1samp(rvn1[:, :, :], 1, axis=2)
    t3, p3 = stats.ttest_1samp(rvn1[0, 0, :], 1)
    assert_array_almost_equal(t1, t2, decimal=14)
    assert_almost_equal(t1[0, 0], t3, decimal=14)
    assert_equal(t1.shape, (n1, n2))
    t, p = stats.ttest_1samp([0, 0, 0], 1)
    assert_equal((np.abs(t), p), (np.inf, 0))

    def convert(t, p, alt):
        if t < 0 and alt == 'less' or (t > 0 and alt == 'greater'):
            return p / 2
        return 1 - p / 2
    converter = np.vectorize(convert)
    tr, pr = stats.ttest_1samp(rvn1[:, :, :], 1)
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, alternative='greater')
    pc = converter(tr, pr, 'greater')
    assert_allclose(p, pc)
    assert_allclose(t, tr)
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, alternative='less')
    pc = converter(tr, pr, 'less')
    assert_allclose(p, pc)
    assert_allclose(t, tr)
    with np.errstate(all='ignore'):
        assert_equal(stats.ttest_1samp([0, 0, 0], 0), (np.nan, np.nan))
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_1samp(anan, 0), ([0, np.nan], [1, np.nan]))
    rvn1[0:2, 1:3, 4:8] = np.nan
    tr, pr = stats.ttest_1samp(rvn1[:, :, :], 1, nan_policy='omit')
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, nan_policy='omit', alternative='greater')
    pc = converter(tr, pr, 'greater')
    assert_allclose(p, pc)
    assert_allclose(t, tr)
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, nan_policy='omit', alternative='less')
    pc = converter(tr, pr, 'less')
    assert_allclose(p, pc)
    assert_allclose(t, tr)