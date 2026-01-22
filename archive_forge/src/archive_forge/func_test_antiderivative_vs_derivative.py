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
def test_antiderivative_vs_derivative(self):
    np.random.seed(1234)
    x = np.linspace(0, 1, 30) ** 2
    y = np.random.rand(len(x))
    spl = splrep(x, y, s=0, k=5)
    pp = PPoly.from_spline(spl)
    for dx in range(0, 10):
        ipp = pp.antiderivative(dx)
        pp2 = ipp.derivative(dx)
        assert_allclose(pp.c, pp2.c)
        for k in range(dx):
            pp2 = ipp.derivative(k)
            r = 1e-13
            endpoint = r * pp2.x[:-1] + (1 - r) * pp2.x[1:]
            assert_allclose(pp2(pp2.x[1:]), pp2(endpoint), rtol=1e-07, err_msg='dx=%d k=%d' % (dx, k))