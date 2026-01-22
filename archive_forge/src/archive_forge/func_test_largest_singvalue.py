import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_largest_singvalue(self) -> None:
    np.random.seed(3)
    rows, cols = (3, 4)
    A = np.random.randn(rows, cols)
    A_sv = np.linalg.svd(A, compute_uv=False)
    X = cp.Variable(shape=(rows, cols))
    sigma = cp.suppfunc(X, [cp.sigma_max(X) <= 1])
    Y = cp.Variable(shape=(rows, cols))
    cons = [Y == A]
    prob = cp.Problem(cp.Minimize(sigma(Y)), cons)
    prob.solve(solver='SCS', eps=1e-08)
    actual = prob.value
    expect = np.sum(A_sv)
    self.assertLessEqual(abs(actual - expect), 1e-06)