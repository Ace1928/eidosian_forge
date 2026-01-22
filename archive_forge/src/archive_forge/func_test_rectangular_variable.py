import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_rectangular_variable(self) -> None:
    np.random.seed(2)
    rows, cols = (4, 2)
    a = np.random.randn(rows, cols)
    x = cp.Variable(shape=(rows, cols))
    sigma = cp.suppfunc(x, [x[:, 0] == 0])
    y = cp.Variable(shape=(rows, cols))
    cons = [sigma(y - a) <= 0]
    objective = cp.Minimize(cp.sum_squares(y.flatten()))
    prob = cp.Problem(objective, cons)
    prob.solve(solver='ECOS')
    expect = np.hstack([np.zeros(shape=(rows, 1)), a[:, [1]]])
    actual = y.value
    self.assertLessEqual(np.linalg.norm(actual - expect, ord=2), 1e-06)
    viol = cons[0].violation()
    self.assertLessEqual(viol, 1e-06)