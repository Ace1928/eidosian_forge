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
def test_logical_indices(self) -> None:
    """Test indexing with boolean arrays.
        """
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    C = Constant(A)
    expr = C[A <= 2]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[A <= 2], expr.value)
    expr = C[A % 2 == 0]
    self.assertEqual(expr.shape, (6,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[A % 2 == 0], expr.value)
    expr = C[np.array([True, False, True]), 3]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[np.array([True, False, True]), 3], expr.value)
    expr = C[1, np.array([True, False, False, True])]
    self.assertEqual(expr.shape, (2,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[1, np.array([True, False, False, True])], expr.value)
    expr = C[np.array([True, True, True]), 1:3]
    self.assertEqual(expr.shape, (3, 2))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[np.array([True, True, True]), 1:3], expr.value)
    expr = C[1:-1, np.array([True, False, True, True])]
    self.assertEqual(expr.shape, (1, 3))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[1:-1, np.array([True, False, True, True])], expr.value)
    expr = C[np.array([True, True, True]), np.array([True, False, True, True])]
    self.assertEqual(expr.shape, (3,))
    self.assertEqual(expr.sign, s.NONNEG)
    self.assertItemsAlmostEqual(A[np.array([True, True, True]), np.array([True, False, True, True])], expr.value)