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
def test_describe_numbers(self):
    x = np.vstack((np.ones((3, 4)), np.full((2, 4), 2)))
    nc, mmc = (5, ([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]))
    mc = np.array([1.4, 1.4, 1.4, 1.4])
    vc = np.array([0.3, 0.3, 0.3, 0.3])
    skc = [0.4082482904638636] * 4
    kurtc = [-1.833333333333333] * 4
    n, mm, m, v, sk, kurt = stats.describe(x)
    assert_equal(n, nc)
    assert_equal(mm, mmc)
    assert_equal(m, mc)
    assert_equal(v, vc)
    assert_array_almost_equal(sk, skc, decimal=13)
    assert_array_almost_equal(kurt, kurtc, decimal=13)
    n, mm, m, v, sk, kurt = stats.describe(x.T, axis=1)
    assert_equal(n, nc)
    assert_equal(mm, mmc)
    assert_equal(m, mc)
    assert_equal(v, vc)
    assert_array_almost_equal(sk, skc, decimal=13)
    assert_array_almost_equal(kurt, kurtc, decimal=13)
    x = np.arange(10.0)
    x[9] = np.nan
    nc, mmc = (9, (0.0, 8.0))
    mc = 4.0
    vc = 7.5
    skc = 0.0
    kurtc = -1.2300000000000002
    n, mm, m, v, sk, kurt = stats.describe(x, nan_policy='omit')
    assert_equal(n, nc)
    assert_equal(mm, mmc)
    assert_equal(m, mc)
    assert_equal(v, vc)
    assert_array_almost_equal(sk, skc)
    assert_array_almost_equal(kurt, kurtc, decimal=13)
    assert_raises(ValueError, stats.describe, x, nan_policy='raise')
    assert_raises(ValueError, stats.describe, x, nan_policy='foobar')