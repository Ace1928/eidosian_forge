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
def test_zeros(self):
    xi = [0, 1, 2, 3]
    yi = [[0, 0], [0], [0, 0], [0, 0]]
    pp = BPoly.from_derivatives(xi, yi)
    assert_(pp.c.shape == (4, 3))
    ppd = pp.derivative()
    for xp in [0.0, 0.1, 1.0, 1.1, 1.9, 2.0, 2.5]:
        assert_allclose([pp(xp), ppd(xp)], [0.0, 0.0])