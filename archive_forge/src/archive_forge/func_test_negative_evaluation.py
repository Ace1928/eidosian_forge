import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_negative_evaluation(self):
    lats = np.array([25, 30, 35, 40, 45])
    lons = np.array([-90, -85, -80, -75, 70])
    mesh = np.meshgrid(lats, lons)
    data = mesh[0] + mesh[1]
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    interpolator = RectSphereBivariateSpline(lat_r, lon_r, data)
    query_lat = np.radians(np.array([35, 37.5]))
    query_lon = np.radians(np.array([-80, -77.5]))
    data_interp = interpolator(query_lat, query_lon)
    ans = np.array([[-45.0, -42.480862], [-49.0625, -46.54315]])
    assert_array_almost_equal(data_interp, ans)