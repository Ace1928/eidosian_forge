import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_jn_yn_cross_product_1(self):
    n = np.array([1, 5, 8])
    x = np.array([0.1, 1, 10])
    left = spherical_jn(n + 1, x) * spherical_yn(n, x) - spherical_jn(n, x) * spherical_yn(n + 1, x)
    right = 1 / x ** 2
    assert_allclose(left, right)