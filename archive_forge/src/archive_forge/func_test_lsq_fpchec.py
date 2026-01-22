import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_lsq_fpchec(self):
    xs = np.arange(100) * 1.0
    ys = np.arange(100) * 1.0
    knots = np.linspace(0, 99, 10)
    bbox = (-1, 101)
    assert_raises(ValueError, LSQUnivariateSpline, xs, ys, knots, bbox=bbox)