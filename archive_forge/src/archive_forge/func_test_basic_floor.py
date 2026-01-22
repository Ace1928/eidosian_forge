import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_floor(self) -> None:
    x = cp.Variable()
    expr = cp.floor(x)
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconvex())
    self.assertTrue(expr.is_quasiconcave())
    self.assertFalse(expr.is_convex())
    self.assertFalse(expr.is_concave())
    self.assertFalse(expr.is_dcp())
    self.assertFalse(expr.is_dgp())
    problem = cp.Problem(cp.Minimize(expr), [x >= 11.8, x <= 17])
    self.assertTrue(problem.is_dqcp())
    self.assertFalse(problem.is_dcp())
    self.assertFalse(problem.is_dgp())
    problem.solve(SOLVER, qcp=True)
    self.assertEqual(problem.objective.value, 11.0)
    self.assertGreater(x.value, 11.7)