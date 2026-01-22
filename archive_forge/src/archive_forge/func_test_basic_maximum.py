import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_maximum(self) -> None:
    x, y = cp.Variable(2)
    expr = cp.maximum(cp.ceil(x), cp.ceil(y))
    problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17, y >= 17.4])
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertEqual(problem.objective.value, 18.0)
    self.assertLess(x.value, 17.1)
    self.assertGreater(x.value, 11.9)
    self.assertGreater(y.value, 17.3)