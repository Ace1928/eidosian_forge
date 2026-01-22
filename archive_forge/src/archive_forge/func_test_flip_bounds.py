import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_flip_bounds(self) -> None:
    x = cp.Variable(pos=True)
    problem = cp.Problem(cp.Maximize(cp.ceil(x)), [x <= 1])
    problem.solve(SOLVER, qcp=True, low=0, high=0.5)
    self.assertGreater(x.value, 0)
    self.assertLessEqual(x.value, 1)
    problem.solve(SOLVER, qcp=True, low=0, high=None)
    self.assertGreater(x.value, 0)
    self.assertLessEqual(x.value, 1)
    problem.solve(SOLVER, qcp=True, low=None, high=0.5)
    self.assertGreater(x.value, 0)
    self.assertLessEqual(x.value, 1)