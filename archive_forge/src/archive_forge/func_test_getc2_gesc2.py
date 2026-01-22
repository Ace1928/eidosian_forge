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
def test_getc2_gesc2():
    np.random.seed(42)
    n = 10
    desired_real = np.random.rand(n)
    desired_cplx = np.random.rand(n) + np.random.rand(n) * 1j
    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            A = np.random.rand(n, n)
            A = A.astype(dtype)
            b = A @ desired_real
            b = b.astype(dtype)
        else:
            A = np.random.rand(n, n) + np.random.rand(n, n) * 1j
            A = A.astype(dtype)
            b = A @ desired_cplx
            b = b.astype(dtype)
        getc2 = get_lapack_funcs('getc2', dtype=dtype)
        gesc2 = get_lapack_funcs('gesc2', dtype=dtype)
        lu, ipiv, jpiv, info = getc2(A, overwrite_a=0)
        x, scale = gesc2(lu, b, ipiv, jpiv, overwrite_rhs=0)
        if ind < 2:
            assert_array_almost_equal(desired_real.astype(dtype), x / scale, decimal=4)
        else:
            assert_array_almost_equal(desired_cplx.astype(dtype), x / scale, decimal=4)