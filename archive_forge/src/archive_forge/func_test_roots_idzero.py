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
def test_roots_idzero(self):
    c = np.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T
    x = np.array([0, 0.4, 0.6, 1.0])
    pp = PPoly(c, x)
    assert_array_equal(pp.roots(), [0.25, 0.4, np.nan, 0.6 + 0.25])
    const = 2.0
    c1 = c.copy()
    c1[1, :] += const
    pp1 = PPoly(c1, x)
    assert_array_equal(pp1.solve(const), [0.25, 0.4, np.nan, 0.6 + 0.25])