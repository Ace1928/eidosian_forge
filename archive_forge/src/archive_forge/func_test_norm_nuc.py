from __future__ import division
import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize
def test_norm_nuc(self) -> None:
    """Test gradient for norm_nuc
        """
    expr = cp.normNuc(self.A)
    self.A.value = [[10, 4], [4, 30]]
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 1])