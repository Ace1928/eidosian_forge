import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_add_canon(self) -> None:
    X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    Y = cvxpy.Constant(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    Z = X + Y
    canon_matrix, constraints = dgp_atom_canon.add_canon(Z, Z.args)
    self.assertEqual(len(constraints), 0)
    self.assertEqual(canon_matrix.shape, Z.shape)
    expected = np.log(np.exp(X.value) + np.exp(Y.value))
    np.testing.assert_almost_equal(expected, canon_matrix.value)
    X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = cvxpy.Constant(2.0)
    Z = X + y
    canon_matrix, constraints = dgp_atom_canon.add_canon(Z, Z.args)
    self.assertEqual(len(constraints), 0)
    self.assertEqual(canon_matrix.shape, Z.shape)
    expected = np.log(np.exp(X.value) + np.exp(y.value))
    np.testing.assert_almost_equal(expected, canon_matrix.value)