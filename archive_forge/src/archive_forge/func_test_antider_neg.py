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
def test_antider_neg(self):
    c = [[1]]
    x = [0, 1]
    b = BPoly(c, x)
    xx = np.linspace(0, 1, 21)
    assert_allclose(b.derivative(-1)(xx), b.antiderivative()(xx), atol=1e-12, rtol=1e-12)
    assert_allclose(b.derivative(1)(xx), b.antiderivative(-1)(xx), atol=1e-12, rtol=1e-12)