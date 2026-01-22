import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_analytic_param_in_exponent(self) -> None:
    base = 2.0
    alpha = cp.Parameter()
    x = cp.Variable(pos=True)
    objective = cp.Maximize(x)
    constr = [cp.one_minus_pos(x) >= cp.Constant(base) ** alpha]
    problem = cp.Problem(objective, constr)
    alpha.value = -1.0
    alpha.delta = 1e-05
    problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-06)
    self.assertAlmostEqual(x.value, 1 - base ** (-1.0))
    problem.backward()
    problem.derivative()
    self.assertAlmostEqual(alpha.gradient, -np.log(base) * base ** (-1.0))
    self.assertAlmostEqual(x.delta, alpha.gradient * 1e-05, places=3)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    alpha.value = -1.2
    alpha.delta = 1e-05
    problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-06)
    self.assertAlmostEqual(x.value, 1 - base ** (-1.2))
    problem.backward()
    problem.derivative()
    self.assertAlmostEqual(alpha.gradient, -np.log(base) * base ** (-1.2))
    self.assertAlmostEqual(x.delta, alpha.gradient * 1e-05, places=3)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)