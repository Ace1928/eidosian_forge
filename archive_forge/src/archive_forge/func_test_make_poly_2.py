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
def test_make_poly_2(self):
    c1 = BPoly._construct_from_derivatives(0, 1, [1, 0], [1])
    assert_allclose(c1, [1.0, 1.0, 1.0])
    c2 = BPoly._construct_from_derivatives(0, 1, [2, 3], [1])
    assert_allclose(c2, [2.0, 7.0 / 2, 1.0])
    c3 = BPoly._construct_from_derivatives(0, 1, [2], [1, 3])
    assert_allclose(c3, [2.0, -0.5, 1.0])