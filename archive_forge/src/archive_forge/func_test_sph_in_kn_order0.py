import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_sph_in_kn_order0(self):
    x = 1.0
    sph_i0 = np.empty((2,))
    sph_i0[0] = spherical_in(0, x)
    sph_i0[1] = spherical_in(0, x, derivative=True)
    sph_i0_expected = np.array([np.sinh(x) / x, np.cosh(x) / x - np.sinh(x) / x ** 2])
    assert_array_almost_equal(r_[sph_i0], sph_i0_expected)
    sph_k0 = np.empty((2,))
    sph_k0[0] = spherical_kn(0, x)
    sph_k0[1] = spherical_kn(0, x, derivative=True)
    sph_k0_expected = np.array([0.5 * pi * exp(-x) / x, -0.5 * pi * exp(-x) * (1 / x + 1 / x ** 2)])
    assert_array_almost_equal(r_[sph_k0], sph_k0_expected)