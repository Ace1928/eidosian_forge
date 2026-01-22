import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_forget_requires_grad(self) -> None:
    np.random.seed(0)
    m, n = (20, 5)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(obj))
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    problem.solve(cp.SCS)
    with self.assertRaisesRegex(ValueError, 'backward can only be called after calling solve with `requires_grad=True`'):
        problem.backward()
    with self.assertRaisesRegex(ValueError, 'derivative can only be called after calling solve with `requires_grad=True`'):
        problem.derivative()