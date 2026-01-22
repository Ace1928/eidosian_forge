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
def test_antider_continuous(self):
    np.random.seed(1234)
    x = np.sort(np.random.random(11))
    c = np.random.random((4, 10))
    bp = BPoly(c, x).antiderivative()
    xx = bp.x[1:-1]
    assert_allclose(bp(xx - 1e-14), bp(xx + 1e-14), atol=1e-12, rtol=1e-12)