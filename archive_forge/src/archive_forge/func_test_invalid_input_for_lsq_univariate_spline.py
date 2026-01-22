import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_invalid_input_for_lsq_univariate_spline(self):
    x_values = [1, 2, 4, 6, 8.5]
    y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
    spl = UnivariateSpline(x_values, y_values, check_finite=True)
    t_values = spl.get_knots()[3:4]
    with assert_raises(ValueError) as info:
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5]
        LSQUnivariateSpline(x_values, y_values, t_values)
    assert 'x and y should have a same length' in str(info.value)
    with assert_raises(ValueError) as info:
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
        w_values = [1.0, 1.0, 1.0, 1.0]
        LSQUnivariateSpline(x_values, y_values, t_values, w=w_values)
    assert 'x, y, and w should have a same length' in str(info.value)
    message = 'Interior knots t must satisfy Schoenberg-Whitney conditions'
    with assert_raises(ValueError, match=message) as info:
        bbox = (100, -100)
        LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
    with assert_raises(ValueError) as info:
        bbox = -1
        LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
    assert 'bbox shape should be (2,)' in str(info.value)
    with assert_raises(ValueError) as info:
        LSQUnivariateSpline(x_values, y_values, t_values, k=6)
    assert 'k should be 1 <= k <= 5' in str(info.value)