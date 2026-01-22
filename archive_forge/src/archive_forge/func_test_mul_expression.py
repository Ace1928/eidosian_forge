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
def test_mul_expression(self) -> None:
    c = Constant([[2], [2]])
    exp = c @ self.x
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual((c[0] @ self.x).sign, s.UNKNOWN)
    self.assertEqual(exp.shape, (1,))
    with self.assertRaises(ValueError):
        [2, 2, 3] @ self.x
    with self.assertRaises(ValueError):
        Constant([[2, 1], [2, 2]]) @ self.C
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = self.A @ self.B
        self.assertTrue(q.is_quadratic())
    T = Constant([[1, 2, 3], [3, 5, 5]])
    exp = (T + T) @ self.B
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (3, 2))
    c = Constant([[2], [2], [-2]])
    exp = [[1], [2]] + c @ self.C
    self.assertEqual(exp.sign, s.UNKNOWN)
    c = Constant([[2], [2]])
    with warnings.catch_warnings(record=True) as w:
        c * self.x
        self.assertEqual(2, len(w))
        self.assertEqual(w[0].category, UserWarning)
        self.assertEqual(w[1].category, DeprecationWarning)
        c * self.x
        self.assertEqual(4, len(w))
        self.assertEqual(w[2].category, UserWarning)
        self.assertEqual(w[3].category, DeprecationWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        c * self.x
        self.assertEqual(5, len(w))
        warnings.simplefilter('ignore', UserWarning)
        c * self.x
        self.assertEqual(len(w), 5)
        warnings.simplefilter('error', UserWarning)
        with self.assertRaises(UserWarning):
            c * self.x