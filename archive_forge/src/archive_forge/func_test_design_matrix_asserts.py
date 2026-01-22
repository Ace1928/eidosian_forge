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
def test_design_matrix_asserts(self):
    np.random.seed(1234)
    n = 10
    k = 3
    x = np.sort(np.random.random_sample(n) * 40 - 20)
    y = np.random.random_sample(n) * 40 - 20
    bspl = make_interp_spline(x, y, k=k)
    with assert_raises(ValueError):
        BSpline.design_matrix(x, bspl.t[::-1], k)
    k = 2
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = [1.0, 2.0, 3.0, 4.0]
    with assert_raises(ValueError):
        BSpline.design_matrix(x, t, k)