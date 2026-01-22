import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_selector_list_indices(self) -> None:
    """Test indexing with lists/ndarrays of indices.
        """
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    C = Constant(A)
    expr = C[[1, 2]]
    self.assertEqual(expr.shape, (2, 4))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[[1, 2]], expr.value)
    expr = C[[0, 2], 3]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[[0, 2], 3], expr.value)
    expr = C[1, [0, 2]]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[1, [0, 2]], expr.value)
    expr = C[[0, 2], 1:3]
    self.assertEqual(expr.shape, (2, 2))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[[0, 2], 1:3], expr.value)
    expr = C[1:-1, [0, 2]]
    self.assertEqual(expr.shape, (1, 2))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[1:-1, [0, 2]], expr.value)
    expr = C[[0, 1], [1, 3]]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[[0, 1], [1, 3]], expr.value)
    expr = C[np.array([0, 1]), [1, 3]]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[np.array([0, 1]), [1, 3]], expr.value)
    expr = C[np.array([0, 1]), np.array([1, 3])]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[np.array([0, 1]), np.array([1, 3])], expr.value)