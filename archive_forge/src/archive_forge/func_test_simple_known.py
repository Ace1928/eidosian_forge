import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
def test_simple_known(self):
    for order in ['C', 'F']:
        A = np.array([[2, 1], [0, 1.0]], order=order)
        LU, P = lu_factor(A)
        assert_allclose(LU, np.array([[2, 1], [0, 1]]))
        assert_array_equal(P, np.array([0, 1]))