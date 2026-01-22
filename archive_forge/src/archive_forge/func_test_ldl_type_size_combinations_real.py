from numpy.testing import assert_array_almost_equal, assert_allclose, assert_
from numpy import (array, eye, zeros, empty_like, empty, tril_indices_from,
from numpy.random import rand, randint, seed
from scipy.linalg import ldl
from scipy._lib._util import ComplexWarning
import pytest
from pytest import raises as assert_raises, warns
@pytest.mark.parametrize('dtype', [float32, float64])
@pytest.mark.parametrize('n', [30, 150])
def test_ldl_type_size_combinations_real(n, dtype):
    seed(1234)
    msg = f'Failed for size: {n}, dtype: {dtype}'
    x = rand(n, n).astype(dtype)
    x = x + x.T
    x += eye(n, dtype=dtype) * dtype(randint(5, 1000000.0))
    l, d1, p = ldl(x)
    u, d2, p = ldl(x, lower=0)
    rtol = 0.0001 if dtype is float32 else 1e-10
    assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg)
    assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg)