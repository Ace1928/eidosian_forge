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
def test_constant(self):
    x = np.ones((7, 4))
    assert_equal(stats.iqr(x), 0.0)
    assert_array_equal(stats.iqr(x, axis=0), np.zeros(4))
    assert_array_equal(stats.iqr(x, axis=1), np.zeros(7))
    assert_equal(stats.iqr(x, interpolation='linear'), 0.0)
    assert_equal(stats.iqr(x, interpolation='midpoint'), 0.0)
    assert_equal(stats.iqr(x, interpolation='nearest'), 0.0)
    assert_equal(stats.iqr(x, interpolation='lower'), 0.0)
    assert_equal(stats.iqr(x, interpolation='higher'), 0.0)
    y = np.ones((4, 5, 6)) * np.arange(6)
    assert_array_equal(stats.iqr(y, axis=0), np.zeros((5, 6)))
    assert_array_equal(stats.iqr(y, axis=1), np.zeros((4, 6)))
    assert_array_equal(stats.iqr(y, axis=2), np.full((4, 5), 2.5))
    assert_array_equal(stats.iqr(y, axis=(0, 1)), np.zeros(6))
    assert_array_equal(stats.iqr(y, axis=(0, 2)), np.full(5, 3.0))
    assert_array_equal(stats.iqr(y, axis=(1, 2)), np.full(4, 3.0))