import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_in_recurrence_complex(self):
    n = np.array([1, 2, 3, 7, 12])
    x = 1.1 + 1.5j
    assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1, x), (2 * n + 1) / x * spherical_in(n, x))