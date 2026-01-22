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
@pytest.mark.parametrize('dtype', DTYPES)
def test_gees_trexc(dtype):
    seed(1234)
    atol = np.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype=dtype)
    gees, trexc = get_lapack_funcs(('gees', 'trexc'), dtype=dtype)
    result = gees(lambda x: None, a, overwrite_a=False)
    assert_equal(result[-1], 0)
    t = result[0]
    z = result[-3]
    d2 = t[6, 6]
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)
    result = trexc(t, z, 7, 1)
    assert_equal(result[-1], 0)
    t = result[0]
    z = result[-2]
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(t[0, 0], d2, rtol=0, atol=atol)