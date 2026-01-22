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
@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_gejsv_edge_arguments(dtype):
    """Test edge arguments return expected status"""
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)
    sva, u, v, work, iwork, info = gejsv(1.0)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    assert_equal(sva, np.array([1.0], dtype=dtype))
    A = np.ones((1,), dtype=dtype)
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    assert_equal(sva, np.array([1.0], dtype=dtype))
    A = np.ones((1, 0), dtype=dtype)
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 0))
    assert_equal(v.shape, (1, 0))
    assert_equal(sva, np.array([], dtype=dtype))
    A = np.sin(np.arange(100).reshape(10, 10)).astype(dtype)
    A = np.asfortranarray(A + A.T)
    Ac = A.copy('A')
    _ = gejsv(A)
    assert_allclose(A, Ac)