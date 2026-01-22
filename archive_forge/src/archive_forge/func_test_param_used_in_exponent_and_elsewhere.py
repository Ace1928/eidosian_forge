import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_param_used_in_exponent_and_elsewhere(self) -> None:
    base = 0.3
    alpha = cp.Parameter(pos=True, value=0.5)
    x = cp.Variable(pos=True)
    objective = cp.Maximize(x)
    constr = [cp.one_minus_pos(x) >= cp.Constant(base) ** alpha + alpha ** 2]
    problem = cp.Problem(objective, constr)
    alpha.delta = 1e-05
    problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-05)
    self.assertAlmostEqual(x.value, 1 - base ** 0.5 - 0.5 ** 2)
    problem.backward()
    problem.derivative()
    self.assertAlmostEqual(alpha.gradient, -np.log(base) * base ** 0.5 - 2 * 0.5)
    self.assertAlmostEqual(x.delta, alpha.gradient * 1e-05, places=3)