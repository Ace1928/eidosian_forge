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
def test_matmul_expression(self) -> None:
    """Test matmul function, corresponding to .__matmul__( operator.
        """
    c = Constant([[2], [2]])
    exp = c.__matmul__(self.x)
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.sign, s.UNKNOWN)
    self.assertEqual(exp.shape, (1,))
    with self.assertRaises(Exception) as cm:
        self.x.__matmul__(2)
    self.assertEqual(str(cm.exception), "Scalar operands are not allowed, use '*' instead")
    with self.assertRaises(ValueError) as cm:
        self.x.__matmul__(np.array([2, 2, 3]))
    with self.assertRaises(Exception) as cm:
        Constant([[2, 1], [2, 2]]).__matmul__(self.C)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = self.A.__matmul__(self.B)
        self.assertTrue(q.is_quadratic())
    T = Constant([[1, 2, 3], [3, 5, 5]])
    exp = (T + T).__matmul__(self.B)
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (3, 2))
    c = Constant([[2], [2], [-2]])
    exp = [[1], [2]] + c.__matmul__(self.C)
    self.assertEqual(exp.sign, s.UNKNOWN)
    a = Parameter((1,))
    x = Variable(shape=(1,))
    expr = a.__matmul__(x)
    self.assertEqual(expr.shape, ())
    a = Parameter((1,))
    x = Variable(shape=(1,))
    expr = a.__matmul__(x)
    self.assertEqual(expr.shape, ())
    A = Parameter((4, 4))
    z = Variable((4, 1))
    expr = A.__matmul__(z)
    self.assertEqual(expr.shape, (4, 1))
    v = Variable((1, 1))
    col_scalar = Parameter((1, 1))
    assert v.shape == col_scalar.shape == col_scalar.T.shape