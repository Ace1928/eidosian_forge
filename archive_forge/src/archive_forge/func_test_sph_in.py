import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_sph_in(self):
    i1n = np.empty((2, 2))
    x = 0.2
    i1n[0][0] = spherical_in(0, x)
    i1n[0][1] = spherical_in(1, x)
    i1n[1][0] = spherical_in(0, x, derivative=True)
    i1n[1][1] = spherical_in(1, x, derivative=True)
    inp0 = i1n[0][1]
    inp1 = i1n[0][0] - 2.0 / 0.2 * i1n[0][1]
    assert_array_almost_equal(i1n[0], np.array([1.00668001270547, 0.06693371456802955]), 12)
    assert_array_almost_equal(i1n[1], [inp0, inp1], 12)