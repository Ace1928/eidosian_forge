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
def test_tpqrt_tpmqrt(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
            B = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
        else:
            A = rand(n, n).astype(dtype)
            B = rand(n, n).astype(dtype)
        tol = 100 * np.spacing(dtype(1.0).real)
        tpqrt, tpmqrt = get_lapack_funcs(('tpqrt', 'tpmqrt'), dtype=dtype)
        for l in (0, n // 2, n):
            a, b, t, info = tpqrt(l, n, A, B)
            assert info == 0
            assert_equal(np.tril(a, -1), np.tril(A, -1))
            assert_equal(np.tril(b, l - n - 1), np.tril(B, l - n - 1))
            B_pent, b_pent = (np.triu(B, l - n), np.triu(b, l - n))
            v = np.concatenate((np.eye(n, dtype=dtype), b_pent))
            Q = np.eye(2 * n, dtype=dtype) - v @ t @ v.T.conj()
            R = np.concatenate((np.triu(a), np.zeros_like(a)))
            assert_allclose(Q.T.conj() @ Q, np.eye(2 * n, dtype=dtype), atol=tol, rtol=0.0)
            assert_allclose(Q @ R, np.concatenate((np.triu(A), B_pent)), atol=tol, rtol=0.0)
            if ind > 1:
                C = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
                D = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
                transpose = 'C'
            else:
                C = rand(n, n).astype(dtype)
                D = rand(n, n).astype(dtype)
                transpose = 'T'
            for side in ('L', 'R'):
                for trans in ('N', transpose):
                    c, d, info = tpmqrt(l, b, t, C, D, side=side, trans=trans)
                    assert info == 0
                    if trans == transpose:
                        q = Q.T.conj()
                    else:
                        q = Q
                    if side == 'L':
                        cd = np.concatenate((c, d), axis=0)
                        CD = np.concatenate((C, D), axis=0)
                        qCD = q @ CD
                    else:
                        cd = np.concatenate((c, d), axis=1)
                        CD = np.concatenate((C, D), axis=1)
                        qCD = CD @ q
                    assert_allclose(cd, qCD, atol=tol, rtol=0.0)
                    if (side, trans) == ('L', 'N'):
                        c_default, d_default, info = tpmqrt(l, b, t, C, D)
                        assert info == 0
                        assert_equal(c_default, c)
                        assert_equal(d_default, d)
            assert_raises(Exception, tpmqrt, l, b, t, C, D, side='A')
            assert_raises(Exception, tpmqrt, l, b, t, C, D, trans='A')