import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_multiply_nonneg(self) -> None:
    x, y = cp.Variable(2, nonneg=True)
    expr = x * y
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconcave())
    self.assertFalse(expr.is_quasiconvex())
    self.assertFalse(expr.is_dcp())
    problem = cp.Problem(cp.Maximize(expr), [x <= 12, y <= 6])
    self.assertTrue(problem.is_dqcp())
    self.assertFalse(problem.is_dcp())
    self.assertFalse(problem.is_dgp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 72, places=1)
    self.assertAlmostEqual(x.value, 12, places=1)
    self.assertAlmostEqual(y.value, 6, places=1)