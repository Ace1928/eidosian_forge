from __future__ import division
import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize
def test_linearize(self) -> None:
    """Test linearize method.
        """
    expr = (2 * self.x - 5)[0]
    self.x.value = [1, 2]
    lin_expr = linearize(expr)
    self.x.value = [55, 22]
    self.assertAlmostEqual(lin_expr.value, expr.value)
    self.x.value = [-1, -5]
    self.assertAlmostEqual(lin_expr.value, expr.value)
    expr = self.A ** 2 + 5
    with self.assertRaises(Exception) as cm:
        linearize(expr)
    self.assertEqual(str(cm.exception), 'Cannot linearize non-affine expression with missing variable values.')
    self.A.value = [[1, 2], [3, 4]]
    lin_expr = linearize(expr)
    manual = expr.value + 2 * cp.reshape(cp.diag(cp.vec(self.A)).value @ cp.vec(self.A - self.A.value), (2, 2))
    self.assertItemsAlmostEqual(lin_expr.value, expr.value)
    self.A.value = [[-5, -5], [8.2, 4.4]]
    assert (lin_expr.value <= expr.value).all()
    self.assertItemsAlmostEqual(lin_expr.value, manual.value)
    expr = cp.log(self.x) / 2
    self.x.value = [1, 2]
    lin_expr = linearize(expr)
    manual = expr.value + cp.diag(0.5 * self.x ** (-1)).value @ (self.x - self.x.value)
    self.assertItemsAlmostEqual(lin_expr.value, expr.value)
    self.x.value = [3, 4.4]
    assert (lin_expr.value >= expr.value).all()
    self.assertItemsAlmostEqual(lin_expr.value, manual.value)