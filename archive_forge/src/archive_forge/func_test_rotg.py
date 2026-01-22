import numpy as np
import pytest
from sklearn.utils._cython_blas import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_rotg(dtype):
    rotg = _rotg_memview[_numpy_to_cython(dtype)]
    rng = np.random.RandomState(0)
    a = dtype(rng.randn())
    b = dtype(rng.randn())
    c, s = (0.0, 0.0)

    def expected_rotg(a, b):
        roe = a if abs(a) > abs(b) else b
        if a == 0 and b == 0:
            c, s, r, z = (1, 0, 0, 0)
        else:
            r = np.sqrt(a ** 2 + b ** 2) * (1 if roe >= 0 else -1)
            c, s = (a / r, b / r)
            z = s if roe == a else 1 if c == 0 else 1 / c
        return (r, z, c, s)
    expected = expected_rotg(a, b)
    actual = rotg(a, b, c, s)
    assert_allclose(actual, expected, rtol=RTOL[dtype])