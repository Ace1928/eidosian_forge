import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
def test_1by1_input_output(self):
    a = self.rng.random([4, 5, 1, 1], dtype=np.float32)
    p, l, u = lu(a, p_indices=True)
    assert_allclose(p, np.zeros(shape=(4, 5, 1), dtype=int))
    assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))
    assert_allclose(u, a)
    a = self.rng.random([4, 5, 1, 1], dtype=np.float32)
    p, l, u = lu(a)
    assert_allclose(p, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))
    assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))
    assert_allclose(u, a)
    pl, u = lu(a, permute_l=True)
    assert_allclose(pl, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))
    assert_allclose(u, a)
    a = self.rng.random([4, 5, 1, 1], dtype=np.float32) * np.complex64(1j)
    p, l, u = lu(a)
    assert_allclose(p, np.ones(shape=(4, 5, 1, 1), dtype=np.complex64))
    assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.complex64))
    assert_allclose(u, a)