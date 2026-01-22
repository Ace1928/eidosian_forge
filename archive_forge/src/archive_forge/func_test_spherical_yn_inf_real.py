import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_yn_inf_real(self):
    n = 6
    x = np.array([-inf, inf])
    assert_allclose(spherical_yn(n, x), np.array([0, 0]))