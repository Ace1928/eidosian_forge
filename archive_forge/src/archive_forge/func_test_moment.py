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
def test_moment(self):
    y = stats.moment(self.scalar_testcase)
    assert_approx_equal(y, 0.0)
    y = stats.moment(self.testcase, 0)
    assert_approx_equal(y, 1.0)
    y = stats.moment(self.testcase, 1)
    assert_approx_equal(y, 0.0, 10)
    y = stats.moment(self.testcase, 2)
    assert_approx_equal(y, 1.25)
    y = stats.moment(self.testcase, 3)
    assert_approx_equal(y, 0.0)
    y = stats.moment(self.testcase, 4)
    assert_approx_equal(y, 2.5625)
    y = stats.moment(self.testcase, [1, 2, 3, 4])
    assert_allclose(y, [0, 1.25, 0, 2.5625])
    y = stats.moment(self.testcase, 0.0)
    assert_approx_equal(y, 1.0)
    assert_raises(ValueError, stats.moment, self.testcase, 1.2)
    y = stats.moment(self.testcase, [1.0, 2, 3, 4.0])
    assert_allclose(y, [0, 1.25, 0, 2.5625])
    message = 'Mean of empty slice\\.|invalid value encountered.*'
    with pytest.warns(RuntimeWarning, match=message):
        y = stats.moment([])
        self._assert_equal(y, np.nan, dtype=np.float64)
        y = stats.moment(np.array([], dtype=np.float32))
        self._assert_equal(y, np.nan, dtype=np.float32)
        y = stats.moment(np.zeros((1, 0)), axis=0)
        self._assert_equal(y, [], shape=(0,), dtype=np.float64)
        y = stats.moment([[]], axis=1)
        self._assert_equal(y, np.nan, shape=(1,), dtype=np.float64)
        y = stats.moment([[]], moment=[0, 1], axis=0)
        self._assert_equal(y, [], shape=(2, 0))
    x = np.arange(10.0)
    x[9] = np.nan
    assert_equal(stats.moment(x, 2), np.nan)
    assert_almost_equal(stats.moment(x, nan_policy='omit'), 0.0)
    assert_raises(ValueError, stats.moment, x, nan_policy='raise')
    assert_raises(ValueError, stats.moment, x, nan_policy='foobar')