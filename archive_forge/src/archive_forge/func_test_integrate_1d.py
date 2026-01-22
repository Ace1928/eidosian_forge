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
def test_integrate_1d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 6, 16, 17, 18)
    x = np.linspace(0, 1, 16 + 1) ** 1
    y = np.linspace(0, 1, 17 + 1) ** 2
    z = np.linspace(0, 1, 18 + 1) ** 3
    p = NdPPoly(c, (x, y, z))
    u = np.random.rand(200)
    v = np.random.rand(200)
    a, b = (0.2, 0.7)
    px = p.integrate_1d(a, b, axis=0)
    pax = p.antiderivative((1, 0, 0))
    assert_allclose(px((u, v)), pax((b, u, v)) - pax((a, u, v)))
    py = p.integrate_1d(a, b, axis=1)
    pay = p.antiderivative((0, 1, 0))
    assert_allclose(py((u, v)), pay((u, b, v)) - pay((u, a, v)))
    pz = p.integrate_1d(a, b, axis=2)
    paz = p.antiderivative((0, 0, 1))
    assert_allclose(pz((u, v)), paz((u, v, b)) - paz((u, v, a)))