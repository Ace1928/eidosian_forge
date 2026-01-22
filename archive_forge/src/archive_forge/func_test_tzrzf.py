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
def test_tzrzf():
    """
    This test performs an RZ decomposition in which an m x n upper trapezoidal
    array M (m <= n) is factorized as M = [R 0] * Z where R is upper triangular
    and Z is unitary.
    """
    seed(1234)
    m, n = (10, 15)
    for ind, dtype in enumerate(DTYPES):
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'), dtype=dtype)
        lwork = _compute_lwork(tzrzf_lw, m, n)
        if ind < 2:
            A = triu(rand(m, n).astype(dtype))
        else:
            A = triu((rand(m, n) + rand(m, n) * 1j).astype(dtype))
        assert_raises(Exception, tzrzf, A.T)
        rz, tau, info = tzrzf(A, lwork=lwork)
        assert_(info == 0)
        R = np.hstack((rz[:, :m], np.zeros((m, n - m), dtype=dtype)))
        V = np.hstack((np.eye(m, dtype=dtype), rz[:, m:]))
        Id = np.eye(n, dtype=dtype)
        ref = [Id - tau[x] * V[[x], :].T.dot(V[[x], :].conj()) for x in range(m)]
        Z = reduce(np.dot, ref)
        assert_allclose(R.dot(Z) - A, zeros_like(A, dtype=dtype), atol=10 * np.spacing(dtype(1.0).real), rtol=0.0)