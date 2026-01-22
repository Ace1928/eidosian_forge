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
def test_tfsm():
    """
    Test for solving a linear system with the coefficient matrix is a
    triangular array stored in Full Packed (RFP) format.
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = triu(rand(n, n) + rand(n, n) * 1j + eye(n)).astype(dtype)
            trans = 'C'
        else:
            A = triu(rand(n, n) + eye(n)).astype(dtype)
            trans = 'T'
        trttf, tfttr, tfsm = get_lapack_funcs(('trttf', 'tfttr', 'tfsm'), dtype=dtype)
        Afp, _ = trttf(A)
        B = rand(n, 2).astype(dtype)
        soln = tfsm(-1, Afp, B)
        assert_array_almost_equal(soln, solve(-A, B), decimal=4 if ind % 2 == 0 else 6)
        soln = tfsm(-1, Afp, B, trans=trans)
        assert_array_almost_equal(soln, solve(-A.conj().T, B), decimal=4 if ind % 2 == 0 else 6)
        A[np.arange(n), np.arange(n)] = dtype(1.0)
        soln = tfsm(-1, Afp, B, trans=trans, diag='U')
        assert_array_almost_equal(soln, solve(-A.conj().T, B), decimal=4 if ind % 2 == 0 else 6)
        B2 = rand(3, n).astype(dtype)
        soln = tfsm(-1, Afp, B2, trans=trans, diag='U', side='R')
        assert_array_almost_equal(soln, solve(-A, B2.T).conj().T, decimal=4 if ind % 2 == 0 else 6)