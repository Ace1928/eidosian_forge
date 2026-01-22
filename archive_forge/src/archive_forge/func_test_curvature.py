import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_curvature(self) -> None:
    x = cp.Variable(3)
    expr = cp.length(x)
    self.assertEqual(expr.curvature, s.QUASICONVEX)
    expr = -cp.length(x)
    self.assertEqual(expr.curvature, s.QUASICONCAVE)
    expr = cp.ceil(x)
    self.assertEqual(expr.curvature, s.QUASILINEAR)
    self.assertTrue(expr.is_quasilinear())