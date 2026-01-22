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
def test_obrientransform():
    x1 = np.array([0, 2, 4])
    t1 = stats.obrientransform(x1)
    expected = [7, -2, 7]
    assert_allclose(t1[0], expected)
    x2 = np.array([0, 3, 6, 9])
    t2 = stats.obrientransform(x2)
    expected = np.array([30, 0, 0, 30])
    assert_allclose(t2[0], expected)
    a, b = stats.obrientransform(x1, x2)
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])
    a, b, c = stats.obrientransform(x1, x2, x1)
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])
    assert_equal(c, t1[0])
    x1 = np.arange(5)
    result = np.array([[5.41666667, 1.04166667, -0.41666667, 1.04166667, 5.41666667], [21.66666667, 4.16666667, -1.66666667, 4.16666667, 21.66666667]])
    assert_array_almost_equal(stats.obrientransform(x1, 2 * x1), result, decimal=8)
    values = range(5, 11)
    reps = np.array([5, 11, 9, 3, 2, 2])
    data = np.repeat(values, reps)
    transformed_values = np.array([3.1828, 0.5591, 0.0344, 1.6086, 5.2817, 11.0538])
    expected = np.repeat(transformed_values, reps)
    result = stats.obrientransform(data)
    assert_array_almost_equal(result[0], expected, decimal=4)