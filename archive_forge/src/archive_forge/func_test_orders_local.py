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
def test_orders_local(self):
    m, k = (7, 12)
    xi, yi = self._make_random_mk(m, k)
    orders = [o + 1 for o in range(m)]
    for i, x in enumerate(xi[1:-1]):
        pp = BPoly.from_derivatives(xi, yi, orders=orders)
        for j in range(orders[i] // 2 + 1):
            assert_allclose(pp(x - 1e-12), pp(x + 1e-12))
            pp = pp.derivative()
        assert_(not np.allclose(pp(x - 1e-12), pp(x + 1e-12)))