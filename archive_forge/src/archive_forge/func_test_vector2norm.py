import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_vector2norm(self) -> None:
    n = 3
    np.random.seed(1)
    a = np.random.randn(n)
    x = cp.Variable(shape=(n,))
    sigma = cp.suppfunc(x, [cp.norm(x - a, 2) <= 1])
    y = np.random.randn(n)
    y_var = cp.Variable(shape=(n,))
    prob = cp.Problem(cp.Minimize(sigma(y_var)), [y == y_var])
    prob.solve(solver='ECOS')
    actual = prob.value
    expected = a @ y + np.linalg.norm(y, ord=2)
    self.assertLessEqual(abs(actual - expected), 1e-06)
    self.assertLessEqual(abs(prob.objective.expr.value - prob.value), 1e-06)