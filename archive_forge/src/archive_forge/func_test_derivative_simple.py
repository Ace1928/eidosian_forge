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
def test_derivative_simple(self):
    np.random.seed(1234)
    c = np.array([[4, 3, 2, 1]]).T
    dc = np.array([[3 * 4, 2 * 3, 2]]).T
    ddc = np.array([[2 * 3 * 4, 1 * 2 * 3]]).T
    x = np.array([0, 1])
    pp = PPoly(c, x)
    dpp = PPoly(dc, x)
    ddpp = PPoly(ddc, x)
    assert_allclose(pp.derivative().c, dpp.c)
    assert_allclose(pp.derivative(2).c, ddpp.c)