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
def test_antiderivative_vs_spline(self):
    np.random.seed(1234)
    x = np.sort(np.r_[0, np.random.rand(11), 1])
    y = np.random.rand(len(x))
    spl = splrep(x, y, s=0, k=5)
    pp = PPoly.from_spline(spl)
    for dx in range(0, 10):
        pp2 = pp.antiderivative(dx)
        spl2 = splantider(spl, dx)
        xi = np.linspace(0, 1, 200)
        assert_allclose(pp2(xi), splev(xi, spl2), rtol=1e-07)