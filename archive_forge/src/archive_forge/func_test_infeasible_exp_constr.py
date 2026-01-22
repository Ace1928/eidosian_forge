import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_infeasible_exp_constr(self) -> None:
    x = cp.Variable()
    constr = [cp.exp(cp.ceil(x)) <= -5]
    problem = cp.Problem(cp.Minimize(0), constr)
    problem.solve(SOLVER, qcp=True)
    self.assertEqual(problem.status, s.INFEASIBLE)