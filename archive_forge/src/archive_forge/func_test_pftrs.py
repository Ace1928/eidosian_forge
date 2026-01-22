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
def test_pftrs():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array and solve a linear system
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
            A = A + A.conj().T + n * eye(n)
        else:
            A = rand(n, n).astype(dtype)
            A = A + A.T + n * eye(n)
        B = ones((n, 3), dtype=dtype)
        Bf1 = ones((n + 2, 3), dtype=dtype)
        Bf2 = ones((n - 2, 3), dtype=dtype)
        pftrs, pftrf, trttf, tfttr = get_lapack_funcs(('pftrs', 'pftrf', 'trttf', 'tfttr'), dtype=dtype)
        Afp, info = trttf(A)
        A_chol_rfp, info = pftrf(n, Afp)
        soln, info = pftrs(n, A_chol_rfp, Bf1)
        assert_(info == 0)
        assert_raises(Exception, pftrs, n, A_chol_rfp, Bf2)
        soln, info = pftrs(n, A_chol_rfp, B)
        assert_(info == 0)
        assert_array_almost_equal(solve(A, B), soln, decimal=4 if ind % 2 == 0 else 6)