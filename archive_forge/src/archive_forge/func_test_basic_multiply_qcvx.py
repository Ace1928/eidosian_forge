import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_multiply_qcvx(self) -> None:
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonpos=True)
    expr = x * y
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconvex())
    self.assertFalse(expr.is_quasiconcave())
    self.assertFalse(expr.is_dcp())
    problem = cp.Problem(cp.Minimize(expr), [x <= 7, y >= -6])
    self.assertTrue(problem.is_dqcp())
    self.assertFalse(problem.is_dcp())
    self.assertFalse(problem.is_dgp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, -42, places=1)
    self.assertAlmostEqual(x.value, 7, places=1)
    self.assertAlmostEqual(y.value, -6, places=1)
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonpos=True)
    expr = y * x
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconvex())
    self.assertFalse(expr.is_quasiconcave())
    self.assertFalse(expr.is_dcp())
    problem = cp.Problem(cp.Minimize(expr), [x <= 7, y >= -6])
    self.assertTrue(problem.is_dqcp())
    self.assertFalse(problem.is_dcp())
    self.assertFalse(problem.is_dgp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, -42, places=1)
    self.assertAlmostEqual(x.value, 7, places=1)
    self.assertAlmostEqual(y.value, -6, places=1)