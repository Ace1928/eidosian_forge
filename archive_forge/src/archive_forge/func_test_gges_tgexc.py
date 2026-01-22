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
def test_gges_tgexc(dtype):
    if dtype == np.float32 and sys.platform == 'darwin' and (blas_provider == 'openblas') and (blas_version < '0.3.21.dev'):
        pytest.xfail('gges[float32] broken for OpenBLAS on macOS, see gh-16949')
    seed(1234)
    atol = np.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)
    gges, tgexc = get_lapack_funcs(('gges', 'tgexc'), dtype=dtype)
    result = gges(lambda x: None, a, b, overwrite_a=False, overwrite_b=False)
    assert_equal(result[-1], 0)
    s = result[0]
    t = result[1]
    q = result[-4]
    z = result[-3]
    d1 = s[0, 0] / t[0, 0]
    d2 = s[6, 6] / t[6, 6]
    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
    result = tgexc(s, t, q, z, 7, 1)
    assert_equal(result[-1], 0)
    s = result[0]
    t = result[1]
    q = result[2]
    z = result[3]
    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
    assert_allclose(s[0, 0] / t[0, 0], d2, rtol=0, atol=atol)
    assert_allclose(s[1, 1] / t[1, 1], d1, rtol=0, atol=atol)