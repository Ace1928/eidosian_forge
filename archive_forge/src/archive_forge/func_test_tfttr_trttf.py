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
def test_tfttr_trttf():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A_full = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
            transr = 'C'
        else:
            A_full = rand(n, n).astype(dtype)
            transr = 'T'
        trttf, tfttr = get_lapack_funcs(('trttf', 'tfttr'), dtype=dtype)
        A_tf_U, info = trttf(A_full)
        assert_(info == 0)
        A_tf_L, info = trttf(A_full, uplo='L')
        assert_(info == 0)
        A_tf_U_T, info = trttf(A_full, transr=transr, uplo='U')
        assert_(info == 0)
        A_tf_L_T, info = trttf(A_full, transr=transr, uplo='L')
        assert_(info == 0)
        A_tf_U_m = zeros((n + 1, n // 2), dtype=dtype)
        A_tf_U_m[:-1, :] = triu(A_full)[:, n // 2:]
        A_tf_U_m[n // 2 + 1:, :] += triu(A_full)[:n // 2, :n // 2].conj().T
        A_tf_L_m = zeros((n + 1, n // 2), dtype=dtype)
        A_tf_L_m[1:, :] = tril(A_full)[:, :n // 2]
        A_tf_L_m[:n // 2, :] += tril(A_full)[n // 2:, n // 2:].conj().T
        assert_array_almost_equal(A_tf_U, A_tf_U_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_U_T, A_tf_U_m.conj().T.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_L, A_tf_L_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_L_T, A_tf_L_m.conj().T.reshape(-1, order='F'))
        A_tr_U, info = tfttr(n, A_tf_U)
        assert_(info == 0)
        A_tr_L, info = tfttr(n, A_tf_L, uplo='L')
        assert_(info == 0)
        A_tr_U_T, info = tfttr(n, A_tf_U_T, transr=transr, uplo='U')
        assert_(info == 0)
        A_tr_L_T, info = tfttr(n, A_tf_L_T, transr=transr, uplo='L')
        assert_(info == 0)
        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_U_T, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))
        assert_array_almost_equal(A_tr_L_T, tril(A_full))