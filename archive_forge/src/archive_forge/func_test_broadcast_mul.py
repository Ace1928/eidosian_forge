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
def test_broadcast_mul(self) -> None:
    """Test multiply broadcasting.
        """
    y = Parameter((3, 1))
    z = Variable((1, 3))
    y.value = np.arange(3)[:, None]
    z.value = (np.arange(3) - 1)[None, :]
    expr = cp.multiply(y, z)
    self.assertItemsAlmostEqual(expr.value, y.value * z.value)
    prob = cp.Problem(cp.Minimize(cp.sum(expr)), [z == z.value])
    prob.solve(solver=cp.SCS)
    self.assertItemsAlmostEqual(expr.value, y.value * z.value)
    np.random.seed(0)
    m, n = (3, 4)
    A = np.random.rand(m, n)
    col_scale = Variable(n)
    with self.assertRaises(ValueError) as cm:
        cp.multiply(A, col_scale)
    self.assertEqual(str(cm.exception), 'Cannot broadcast dimensions  (3, 4) (4,)')
    col_scale = Variable([1, n])
    C = cp.multiply(A, col_scale)
    self.assertEqual(C.shape, (m, n))
    row_scale = Variable([m, 1])
    R = cp.multiply(A, row_scale)
    self.assertEqual(R.shape, (m, n))