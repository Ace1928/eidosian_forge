import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_basic_lmi(self) -> None:
    np.random.seed(4)
    n = 3
    A = np.random.randn(n, n)
    A = A.T @ A
    X = cp.Variable(shape=(n, n))
    sigma = cp.suppfunc(X, [0 << X, cp.lambda_max(X) <= 1])
    Y = cp.Variable(shape=(n, n))
    cons = [Y == A]
    expr = sigma(Y)
    prob = cp.Problem(cp.Minimize(expr), cons)
    prob.solve(solver='SCS', eps=1e-08)
    actual1 = prob.value
    actual2 = expr.value
    self.assertLessEqual(abs(actual1 - actual2), 1e-06)
    expect = np.trace(A)
    self.assertLessEqual(abs(actual1 - expect), 0.0001)