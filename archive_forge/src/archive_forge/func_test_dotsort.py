from __future__ import division
import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize
def test_dotsort(self) -> None:
    """Test dotsort.
        """
    expr = cp.dotsort(self.A, [0.1, -2])
    self.A.value = [[4, 3], [2, 1]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0.1, 0, 0, -2])
    self.A.value = [[1, 2], [3, 0.5]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 0.1, 0, -2])
    expr = cp.dotsort(self.A, [1, 1])
    self.A.value = [[4, 3], [2, 1]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 1, 0])
    self.A.value = [[1, 2], [3, 0.5]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 1, 0])
    expr = -cp.dotsort(self.A, [-1, -1])
    self.A.value = [[4, 3], [2, 1]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 0, 1])
    self.A.value = [[1, 2], [3, 0.5]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 1])