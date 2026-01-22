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
def test_interp2d_eval_unsorted(self):
    y, x = mgrid[0:2:20j, 0:pi:21j]
    z = sin(x + 0.5 * y)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        func = interp2d(x, y, z)
        xe = np.array([3, 4, 5])
        ye = np.array([5.3, 7.1])
        assert_allclose(func(xe, ye), func(xe, ye[::-1]))
        assert_raises(ValueError, func, xe, ye[::-1], 0, 0, True)