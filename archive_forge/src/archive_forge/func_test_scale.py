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
def test_scale(self):
    x = np.arange(15.0).reshape((3, 5))
    assert_equal(stats.iqr(x, scale=1.0), 7)
    assert_almost_equal(stats.iqr(x, scale='normal'), 7 / 1.3489795)
    assert_equal(stats.iqr(x, scale=2.0), 3.5)
    x[1, 2] = np.nan
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert_equal(stats.iqr(x, scale=1.0, nan_policy='propagate'), np.nan)
        assert_equal(stats.iqr(x, scale='normal', nan_policy='propagate'), np.nan)
        assert_equal(stats.iqr(x, scale=2.0, nan_policy='propagate'), np.nan)
        assert_equal(stats.iqr(x, axis=1, scale=1.0, nan_policy='propagate'), [2, np.nan, 2])
        assert_almost_equal(stats.iqr(x, axis=1, scale='normal', nan_policy='propagate'), np.array([2, np.nan, 2]) / 1.3489795)
        assert_equal(stats.iqr(x, axis=1, scale=2.0, nan_policy='propagate'), [1, np.nan, 1])
    assert_equal(stats.iqr(x, scale=1.0, nan_policy='omit'), 7.5)
    assert_almost_equal(stats.iqr(x, scale='normal', nan_policy='omit'), 7.5 / 1.3489795)
    assert_equal(stats.iqr(x, scale=2.0, nan_policy='omit'), 3.75)
    assert_raises(ValueError, stats.iqr, x, scale='foobar')