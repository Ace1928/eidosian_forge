import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_non_dpp_problem_raises_error(self) -> None:
    alpha = cp.Parameter(pos=True, value=1.0)
    x = cp.Variable(pos=True)
    dgp = cp.Problem(cp.Minimize((alpha * x) ** alpha), [x == alpha])
    self.assertTrue(dgp.objective.is_dgp())
    self.assertFalse(dgp.objective.is_dgp(dpp=True))
    with self.assertRaises(error.DPPError):
        dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=False)
        self.assertAlmostEqual(x.value, 1.0)