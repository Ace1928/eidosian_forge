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
def test_design_matrix_x_shapes(self):
    np.random.seed(1234)
    n = 10
    k = 3
    x = np.sort(np.random.random_sample(n) * 40 - 20)
    y = np.random.random_sample(n) * 40 - 20
    bspl = make_interp_spline(x, y, k=k)
    for i in range(1, 4):
        xc = x[:i]
        yc = y[:i]
        des_matr_csr = BSpline.design_matrix(xc, bspl.t, k).toarray()
        assert_allclose(des_matr_csr @ bspl.c, yc, atol=1e-14)