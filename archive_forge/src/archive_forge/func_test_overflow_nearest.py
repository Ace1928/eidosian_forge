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
def test_overflow_nearest(self):
    for kind in ('nearest', 'previous', 'next'):
        x = np.array([0, 50, 127], dtype=np.int8)
        ii = interp1d(x, x, kind=kind)
        assert_array_almost_equal(ii(x), x)