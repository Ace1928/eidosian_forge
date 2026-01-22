import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_Rn(self) -> None:
    np.random.seed(0)
    n = 5
    x = cp.Variable(shape=(n,))
    sigma = cp.suppfunc(x, [])
    a = np.random.randn(n)
    y = cp.Variable(shape=(n,))
    cons = [sigma(y - a) <= 0]
    objective = cp.Minimize(a @ y)
    prob = cp.Problem(objective, cons)
    prob.solve(solver='ECOS')
    actual = prob.value
    expected = np.dot(a, a)
    self.assertLessEqual(abs(actual - expected), 1e-06)
    actual = y.value
    expected = a
    self.assertLessEqual(np.linalg.norm(actual - expected, ord=2), 1e-06)
    viol = cons[0].violation()
    self.assertLessEqual(viol, 1e-08)