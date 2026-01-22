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
def test_simple5(self):
    x = [0, 1]
    c = [[1], [1], [8], [2], [1]]
    bp = BPoly(c, x)
    assert_allclose(bp(0.3), 0.7 ** 4 + 4 * 0.7 ** 3 * 0.3 + 8 * 6 * 0.7 ** 2 * 0.3 ** 2 + 2 * 4 * 0.7 * 0.3 ** 3 + 0.3 ** 4)