import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_invalid_input_2(self):
    data = np.dot(np.atleast_2d(90.0 - np.linspace(-80.0, 80.0, 18)).T, np.atleast_2d(180.0 - np.abs(np.linspace(0.0, 350.0, 9)))).T
    with assert_raises(ValueError) as exc_info:
        lats = np.linspace(0, 170, 9) * np.pi / 180.0
        lons = np.linspace(0, 350, 18) * np.pi / 180.0
        RectSphereBivariateSpline(lats, lons, data)
    assert 'u should be between (0, pi)' in str(exc_info.value)
    with assert_raises(ValueError) as exc_info:
        lats = np.linspace(10, 180, 9) * np.pi / 180.0
        lons = np.linspace(0, 350, 18) * np.pi / 180.0
        RectSphereBivariateSpline(lats, lons, data)
    assert 'u should be between (0, pi)' in str(exc_info.value)
    with assert_raises(ValueError) as exc_info:
        lats = np.linspace(10, 170, 9) * np.pi / 180.0
        lons = np.linspace(-181, 10, 18) * np.pi / 180.0
        RectSphereBivariateSpline(lats, lons, data)
    assert 'v[0] should be between [-pi, pi)' in str(exc_info.value)
    with assert_raises(ValueError) as exc_info:
        lats = np.linspace(10, 170, 9) * np.pi / 180.0
        lons = np.linspace(-10, 360, 18) * np.pi / 180.0
        RectSphereBivariateSpline(lats, lons, data)
    assert 'v[-1] should be v[0] + 2pi or less' in str(exc_info.value)
    with assert_raises(ValueError) as exc_info:
        lats = np.linspace(10, 170, 9) * np.pi / 180.0
        lons = np.linspace(10, 350, 18) * np.pi / 180.0
        RectSphereBivariateSpline(lats, lons, data, s=-1)
    assert 's should be positive' in str(exc_info.value)