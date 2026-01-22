import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_solve(self) -> None:
    x = cp.Variable()
    expr = cp.ceil(x)
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconvex())
    self.assertTrue(expr.is_quasiconcave())
    self.assertFalse(expr.is_convex())
    self.assertFalse(expr.is_concave())
    self.assertFalse(expr.is_dcp())
    self.assertFalse(expr.is_dgp())
    problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17])
    self.assertTrue(problem.is_dqcp())
    self.assertFalse(problem.is_dcp())
    self.assertFalse(problem.is_dgp())
    problem.solve(SOLVER, qcp=True, low=12, high=17)
    self.assertAlmostEqual(problem.value, 12.0, places=3)
    self.assertAlmostEqual(x.value, 12.0, places=3)
    problem._clear_solution()
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.value, 12.0, places=3)
    self.assertAlmostEqual(x.value, 12.0, places=3)
    problem._clear_solution()
    problem.solve(SOLVER, qcp=True, high=17)
    self.assertAlmostEqual(problem.value, 12.0, places=3)
    self.assertAlmostEqual(x.value, 12.0, places=3)
    problem._clear_solution()
    problem.solve(SOLVER, qcp=True, low=12)
    self.assertAlmostEqual(problem.value, 12.0, places=3)
    self.assertAlmostEqual(x.value, 12.0, places=3)
    problem._clear_solution()
    problem.solve(SOLVER, qcp=True, low=0, high=100)
    self.assertAlmostEqual(problem.value, 12.0, places=3)
    self.assertAlmostEqual(x.value, 12.0, places=3)