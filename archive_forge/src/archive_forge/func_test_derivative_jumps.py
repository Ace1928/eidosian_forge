import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_derivative_jumps(self):
    k = 2
    t = [-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7]
    np.random.seed(1234)
    c = np.r_[0, 0, np.random.random(5), 0, 0]
    b = BSpline(t, c, k)
    x = np.asarray([1, 3, 4, 6])
    assert_allclose(b(x[x != 6] - 1e-10), b(x[x != 6] + 1e-10))
    assert_(not np.allclose(b(6.0 - 1e-10), b(6 + 1e-10)))
    x0 = np.asarray([3, 4])
    assert_allclose(b(x0 - 1e-10, nu=1), b(x0 + 1e-10, nu=1))
    x1 = np.asarray([1, 6])
    assert_(not np.all(np.allclose(b(x1 - 1e-10, nu=1), b(x1 + 1e-10, nu=1))))
    assert_(not np.all(np.allclose(b(x - 1e-10, nu=2), b(x + 1e-10, nu=2))))