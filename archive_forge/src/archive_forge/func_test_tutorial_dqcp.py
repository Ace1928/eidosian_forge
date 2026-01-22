import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_tutorial_dqcp(self) -> None:
    x = cp.Variable(nonneg=True)
    concave_frac = x * cp.sqrt(x)
    constraint = [cp.ceil(x) <= 10]
    problem = cp.Problem(cp.Maximize(concave_frac), constraint)
    self.assertTrue(concave_frac.is_quasiconcave())
    self.assertTrue(constraint[0].is_dqcp())
    self.assertTrue(problem.is_dqcp())
    w = cp.Variable()
    fn = w * cp.sqrt(w)
    problem = cp.Problem(cp.Maximize(fn))
    self.assertFalse(fn.is_dqcp())
    self.assertFalse(problem.is_dqcp())