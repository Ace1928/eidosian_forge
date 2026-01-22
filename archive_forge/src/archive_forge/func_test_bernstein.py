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
def test_bernstein(self):
    k = 3
    t = np.asarray([0] * (k + 1) + [1] * (k + 1))
    c = np.asarray([1.0, 2.0, 3.0, 4.0])
    bp = BPoly(c.reshape(-1, 1), [0, 1])
    bspl = BSpline(t, c, k)
    xx = np.linspace(-1.0, 2.0, 10)
    assert_allclose(bp(xx, extrapolate=True), bspl(xx, extrapolate=True), atol=1e-14)
    assert_allclose(splev(xx, (t, c, k)), bspl(xx), atol=1e-14)