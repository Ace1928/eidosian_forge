from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def test_deriv_3d_simple(self):
    c = np.ones((1, 1, 1, 3, 4, 5))
    x = np.linspace(0, 1, 3 + 1) ** 1
    y = np.linspace(0, 1, 4 + 1) ** 2
    z = np.linspace(0, 1, 5 + 1) ** 3
    p = NdPPoly(c, (x, y, z))
    ip = p.antiderivative((1, 0, 4))
    ip = ip.antiderivative((0, 2, 0))
    xi = np.random.rand(20)
    yi = np.random.rand(20)
    zi = np.random.rand(20)
    assert_allclose(ip((xi, yi, zi)), xi * yi ** 2 * zi ** 4 / (gamma(3) * gamma(5)))