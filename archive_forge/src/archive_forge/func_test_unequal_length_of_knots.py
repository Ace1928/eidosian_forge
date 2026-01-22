import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_unequal_length_of_knots(self):
    """Test for the case when the input knot-location arrays in x and y are
        of different lengths.
        """
    x, y = np.mgrid[0:100, 0:100]
    x = x.ravel()
    y = y.ravel()
    z = 3.0 * np.ones_like(x)
    tx = np.linspace(0.1, 98.0, 29)
    ty = np.linspace(0.1, 98.0, 33)
    with suppress_warnings() as sup:
        r = sup.record(UserWarning, '\nThe coefficients of the spline')
        lut = LSQBivariateSpline(x, y, z, tx, ty)
        assert_equal(len(r), 1)
    assert_almost_equal(lut(x, y, grid=False), z)