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
def test_quad_form_matmul(self) -> None:
    """Test conversion of native x.T @ A @ x into QuadForms.
        """
    x = Variable(shape=(2,))
    A = Constant([[1, 0], [0, -1]])
    expr = x.T.__matmul__(A).__matmul__(x)
    assert isinstance(expr, cp.QuadForm)
    x = Variable(shape=(2,))
    A = Constant([[1, 0], [0, -1]])
    expr = 1 / 2 * x.T.__matmul__(A).__matmul__(x) + x.T.__matmul__(x)
    assert isinstance(expr.args[0].args[1], cp.QuadForm)
    assert expr.args[0].args[1].args[0] is x
    x = Variable(shape=(2,))
    A = Constant([[1, 0], [0, -1]])
    c = Constant([2, -2])
    expr = 1 / 2 * c.T.__matmul__(c) * x.T.__matmul__(A).__matmul__(x) + x.T.__matmul__(x)
    assert isinstance(expr.args[0].args[1], cp.QuadForm)
    assert expr.args[0].args[1].args[0] is x
    x = Variable(shape=(2,))
    A = Constant(sp.eye(2))
    expr = x.T.__matmul__(A).__matmul__(x)
    assert isinstance(expr, cp.QuadForm)
    x = Variable(shape=(2,))
    A = Constant(np.eye(3))
    with self.assertRaises(Exception) as _:
        x.T.__matmul__(A).__matmul__(x)
    x = cp.Variable(shape=(2,))
    A = cp.Constant([[1, 0], [0, 1]])
    expr = x.T.__matmul__(psd_wrap(A)).__matmul__(x)
    assert isinstance(expr, cp.QuadForm)
    x = cp.Variable(shape=(2,))
    A = cp.Constant([[2, 0, 0], [0, 0, 1]])
    M = cp.Constant([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    b = cp.Constant([1, 2, 3])
    y = A.__matmul__(x) - b
    expr = y.T.__matmul__(M).__matmul__(y)
    assert isinstance(expr, cp.QuadForm)
    assert expr.args[0] is y
    assert expr.args[1] is M
    x = Variable(shape=(2,))
    A = Parameter(shape=(2, 2), symmetric=True)
    expr = x.T.__matmul__(A).__matmul__(x)
    assert isinstance(expr, cp.QuadForm)
    x = Variable(shape=(2,))
    A = Constant([[1, 0], [1, 1]])
    with self.assertRaises(ValueError) as _:
        x.T.__matmul__(A).__matmul__(x)
    x = Variable(shape=(2,))
    A = Constant([[1, 1j], [1j, 1]])
    with self.assertRaises(ValueError) as _:
        x.T.__matmul__(A).__matmul__(x)
    x = Variable(shape=(2,))
    y = Variable(shape=(2,))
    A = Constant([[1, 0], [0, -1]])
    expr = x.T.__matmul__(A).__matmul__(y)
    assert not isinstance(expr, cp.QuadForm)
    x = Variable(shape=(2,))
    M = Variable(shape=(2, 2))
    expr = x.T.__matmul__(M).__matmul__(x)
    assert not isinstance(expr, cp.QuadForm)
    x = Constant([1, 0])
    M = Variable(shape=(2, 2))
    expr = x.T.__matmul__(M).__matmul__(x)
    assert not isinstance(expr, cp.QuadForm)