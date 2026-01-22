import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_scalar_sum(self) -> None:
    x = cp.Variable(pos=True)
    problem = cp.Problem(cp.Minimize(cp.sum(1 / x)))
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.value, 0, places=3)
    problem = cp.Problem(cp.Minimize(cp.cumsum(1 / x)))
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.value, 0, places=3)