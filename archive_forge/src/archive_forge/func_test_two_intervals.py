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
def test_two_intervals(self):
    x = [0, 1, 3]
    c = [[3, 0], [0, 0], [0, 2]]
    bp = BPoly(c, x)
    assert_allclose(bp(0.4), 3 * 0.6 * 0.6)
    assert_allclose(bp(1.7), 2 * (0.7 / 2) ** 2)