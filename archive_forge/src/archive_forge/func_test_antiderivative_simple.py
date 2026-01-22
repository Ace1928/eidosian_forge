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
def test_antiderivative_simple(self):
    x = [0, 1, 3]
    c = [[0, 0], [1, 1]]
    bp = BPoly(c, x)
    bi = bp.antiderivative()
    xx = np.linspace(0, 3, 11)
    assert_allclose(bi(xx), np.where(xx < 1, xx ** 2 / 2.0, 0.5 * xx * (xx / 2.0 - 1) + 3.0 / 4), atol=1e-12, rtol=1e-12)