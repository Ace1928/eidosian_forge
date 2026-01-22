import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_jn_exact(self):
    x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
    assert_allclose(spherical_jn(2, x), (-1 / x + 3 / x ** 3) * sin(x) - 3 / x ** 2 * cos(x))