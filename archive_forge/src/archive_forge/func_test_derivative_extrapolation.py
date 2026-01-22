import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_derivative_extrapolation(self):
    x_values = [1, 2, 4, 6, 8.5]
    y_values = [0.5, 0.8, 1.3, 2.5, 5]
    f = UnivariateSpline(x_values, y_values, ext='const', k=3)
    x = [-1, 0, -0.5, 9, 9.5, 10]
    assert_allclose(f.derivative()(x), 0, atol=1e-15)