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
def test_orders_too_high(self):
    m, k = (5, 12)
    xi, yi = self._make_random_mk(m, k)
    BPoly.from_derivatives(xi, yi, orders=2 * k - 1)
    assert_raises(ValueError, BPoly.from_derivatives, **dict(xi=xi, yi=yi, orders=2 * k))