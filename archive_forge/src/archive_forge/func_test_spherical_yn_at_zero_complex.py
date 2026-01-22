import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_yn_at_zero_complex(self):
    n = np.array([0, 1, 2, 5, 10, 100])
    x = 0 + 0j
    assert_allclose(spherical_yn(n, x), np.full(n.shape, nan))