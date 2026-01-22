import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_spline_2d_outofbounds(self):
    x = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
    y = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
    z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    lut = RectBivariateSpline(x, y, z)
    xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
    actual = interpn((x, y), z, xi, method='splinef2d', bounds_error=False, fill_value=999.99)
    expected = lut.ev(xi[:, 0], xi[:, 1])
    expected[2:4] = 999.99
    assert_array_almost_equal(actual, expected)
    assert_raises(ValueError, interpn, (x, y), z, xi, method='splinef2d', bounds_error=False, fill_value=None)