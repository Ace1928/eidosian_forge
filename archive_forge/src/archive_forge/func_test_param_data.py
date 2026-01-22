import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def test_param_data(self) -> None:
    for solver in self.solvers:
        np.random.seed(0)
        m = 30
        n = 20
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        gamma_val = 0.5
        gamma_val_new = 0.1
        objective = cp.Minimize(gamma * cp.sum_squares(A @ x - b) + cp.norm(x, 1))
        constraints = [1 <= x, x <= 2]
        prob = cp.Problem(objective, constraints)
        self.assertTrue(prob.is_dpp())
        gamma.value = gamma_val_new
        data_scratch, _, _ = prob.get_problem_data(solver)
        prob.solve(solver=solver)
        x_scratch = np.copy(x.value)
        prob = cp.Problem(objective, constraints)
        gamma.value = gamma_val
        data_param, _, _ = prob.get_problem_data(solver)
        prob.solve(solver=solver)
        gamma.value = gamma_val_new
        data_param_new, _, _ = prob.get_problem_data(solver)
        prob.solve(solver=solver)
        x_gamma_new = np.copy(x.value)
        np.testing.assert_allclose(data_param_new['P'].todense(), data_scratch['P'].todense())
        np.testing.assert_allclose(x_gamma_new, x_scratch, rtol=0.01, atol=0.01)