import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal as np_assert_equal
from ..dwiparams import B2q, q2bg
def test_q2bg():
    for pos in range(3):
        q_vec = np.zeros((3,))
        q_vec[pos] = 10.0
        np_assert_equal(q2bg(q_vec), (10, q_vec / 10.0))
    q_vec = [0, 1e-06, 0]
    np_assert_equal(q2bg(q_vec), (0, 0))
    q_vec = [0, 0.0001, 0]
    b, g = q2bg(q_vec)
    assert_array_almost_equal(b, 0.0001)
    assert_array_almost_equal(g, [0, 1, 0])
    np_assert_equal(q2bg(q_vec, tol=0.0005), (0, 0))