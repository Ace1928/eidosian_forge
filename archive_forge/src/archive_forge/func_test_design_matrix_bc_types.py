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
def test_design_matrix_bc_types(self):
    """
        Splines with different boundary conditions are built on different
        types of vectors of knots. As far as design matrix depends only on
        vector of knots, `k` and `x` it is useful to make tests for different
        boundary conditions (and as following different vectors of knots).
        """

    def run_design_matrix_tests(n, k, bc_type):
        """
            To avoid repetition of code the following function is provided.
            """
        np.random.seed(1234)
        x = np.sort(np.random.random_sample(n) * 40 - 20)
        y = np.random.random_sample(n) * 40 - 20
        if bc_type == 'periodic':
            y[0] = y[-1]
        bspl = make_interp_spline(x, y, k=k, bc_type=bc_type)
        c = np.eye(len(bspl.t) - k - 1)
        des_matr_def = BSpline(bspl.t, c, k)(x)
        des_matr_csr = BSpline.design_matrix(x, bspl.t, k).toarray()
        assert_allclose(des_matr_csr @ bspl.c, y, atol=1e-14)
        assert_allclose(des_matr_def, des_matr_csr, atol=1e-14)
    n = 11
    k = 3
    for bc in ['clamped', 'natural']:
        run_design_matrix_tests(n, k, bc)
    for k in range(3, 8, 2):
        run_design_matrix_tests(n, k, 'not-a-knot')
    n = 5
    for k in range(2, 7):
        run_design_matrix_tests(n, k, 'periodic')