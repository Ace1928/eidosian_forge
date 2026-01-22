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
def test_from_spline(self):
    np.random.seed(1234)
    x = np.sort(np.r_[0, np.random.rand(11), 1])
    y = np.random.rand(len(x))
    spl = splrep(x, y, s=0)
    pp = PPoly.from_spline(spl)
    xi = np.linspace(0, 1, 200)
    assert_allclose(pp(xi), splev(xi, spl))
    b = BSpline(*spl)
    ppp = PPoly.from_spline(b)
    assert_allclose(ppp(xi), b(xi))
    t, c, k = spl
    for extrap in (None, True, False):
        b = BSpline(t, c, k, extrapolate=extrap)
        p = PPoly.from_spline(b)
        assert_equal(p.extrapolate, b.extrapolate)