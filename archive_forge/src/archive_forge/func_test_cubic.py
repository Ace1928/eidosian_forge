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
def test_cubic(self):
    interp10 = interp1d(self.x10, self.y10, kind='cubic')
    assert_array_almost_equal(interp10(self.x10), self.y10)
    assert_array_almost_equal(interp10(1.2), np.array([1.2]))
    assert_array_almost_equal(interp10(1.5), np.array([1.5]))
    assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.4, 5.6, 6.0]))