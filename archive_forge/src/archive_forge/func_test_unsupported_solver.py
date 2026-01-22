import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_unsupported_solver(self) -> None:
    x = cp.Variable()
    param = cp.Parameter()
    problem = cp.Problem(cp.Minimize(x), [x <= param])
    param.value = 1
    with self.assertRaisesRegex(ValueError, 'When requires_grad is True, the only supported solver is SCS.*'):
        problem.solve(cp.ECOS, requires_grad=True)