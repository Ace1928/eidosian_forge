import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_gglse():
    for ind, dtype in enumerate(DTYPES):
        func, func_lwork = get_lapack_funcs(('gglse', 'gglse_lwork'), dtype=dtype)
        lwork = _compute_lwork(func_lwork, m=6, n=4, p=2)
        if ind < 2:
            a = np.array([[-0.57, -1.28, -0.39, 0.25], [-1.93, 1.08, -0.31, -2.14], [2.3, 0.24, 0.4, -0.35], [-1.93, 0.64, -0.66, 0.08], [0.15, 0.3, 0.15, -2.13], [-0.02, 1.03, -1.43, 0.5]], dtype=dtype)
            c = np.array([-1.5, -2.14, 1.23, -0.54, -1.68, 0.82], dtype=dtype)
            d = np.array([0.0, 0.0], dtype=dtype)
        else:
            a = np.array([[0.96 - 0.81j, -0.03 + 0.96j, -0.91 + 2.06j, -0.05 + 0.41j], [-0.98 + 1.98j, -1.2 + 0.19j, -0.66 + 0.42j, -0.81 + 0.56j], [0.62 - 0.46j, 1.01 + 0.02j, 0.63 - 0.17j, -1.11 + 0.6j], [0.37 + 0.38j, 0.19 - 0.54j, -0.98 - 0.36j, 0.22 - 0.2j], [0.83 + 0.51j, 0.2 + 0.01j, -0.17 - 0.46j, 1.47 + 1.59j], [1.08 - 0.28j, 0.2 - 0.12j, -0.07 + 1.23j, 0.26 + 0.26j]])
            c = np.array([[-2.54 + 0.09j], [1.65 - 2.26j], [-2.11 - 3.96j], [1.82 + 3.3j], [-6.41 + 3.77j], [2.07 + 0.66j]])
            d = np.zeros(2, dtype=dtype)
        b = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]], dtype=dtype)
        _, _, _, result, _ = func(a, b, c, d, lwork=lwork)
        if ind < 2:
            expected = np.array([0.48904455, 0.99754786, 0.48904455, 0.99754786])
        else:
            expected = np.array([1.08742917 - 1.96205783j, -0.74093902 + 3.72973919j, 1.08742917 - 1.96205759j, -0.74093896 + 3.72973895j])
        assert_array_almost_equal(result, expected, decimal=4)