import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_nocedal_example(self):
    H = csc_matrix([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
    A = csc_matrix([[1, 0, 1], [0, 1, 1]])
    c = np.array([-8, -3, -3])
    b = -np.array([3, 0])
    Z, _, Y = projections(A)
    x, info = projected_cg(H, c, Z, Y, b)
    assert_equal(info['stop_cond'], 4)
    assert_equal(info['hits_boundary'], False)
    assert_array_almost_equal(x, [2, -1, 1])