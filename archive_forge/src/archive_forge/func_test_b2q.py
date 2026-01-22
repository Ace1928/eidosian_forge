import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal as np_assert_equal
from ..dwiparams import B2q, q2bg
def test_b2q():
    q = np.array([1, 2, 3])
    s = np.sqrt(np.sum(q * q))
    B = np.outer(q, q)
    assert_array_almost_equal(q * s, B2q(B))
    q = np.array([1, 2, 3])
    B = np.outer(-q, -q)
    assert_array_almost_equal(q * s, B2q(B))
    q = np.array([-1, 2, 3])
    B = np.outer(q, q)
    assert_array_almost_equal(-q * s, B2q(B))
    B = np.eye(3) * -1
    with pytest.raises(ValueError):
        B2q(B)
    q = B2q(B, tol=1)
    B = np.diag([-1e-14, 10.0, 1])
    with pytest.raises(ValueError):
        B2q(B)
    assert_array_almost_equal(B2q(B, tol=5e-13), [0, 10, 0])
    B = np.eye(3)
    B[0, 1] = 1e-05
    with pytest.raises(ValueError):
        B2q(B)