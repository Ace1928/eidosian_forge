import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def test_qp_problem(self) -> None:
    for solver in self.solvers:
        m = 30
        n = 20
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 0.5
        objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma * cp.norm(x, 1))
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        x_full = np.copy(x.value)
        solving_chain = problem._cache.solving_chain
        solver = problem._cache.solving_chain.solver
        inverse_data = problem._cache.inverse_data
        param_prog = problem._cache.param_prog
        data, solver_inverse_data = solving_chain.solver.apply(param_prog)
        inverse_data = inverse_data + [solver_inverse_data]
        raw_solution = solver.solve_via_data(data, warm_start=False, verbose=False, solver_opts={})
        problem.unpack_results(raw_solution, solving_chain, inverse_data)
        x_param = np.copy(x.value)
        np.testing.assert_allclose(x_param, x_full, rtol=0.01, atol=0.01)