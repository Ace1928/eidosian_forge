import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_zero_in_problem_data(self) -> None:
    x = cp.Variable()
    param = cp.Parameter()
    param.value = 0.0
    problem = cp.Problem(cp.Minimize(x), [param * x >= 0])
    data, _, _ = problem.get_problem_data(cp.DIFFCP)
    A = data[s.A]
    self.assertIn(0.0, A.data)