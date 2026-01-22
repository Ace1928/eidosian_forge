import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_paper_example_is_dpp(self) -> None:
    F = cp.Parameter((2, 2))
    x = cp.Variable((2, 1))
    g = cp.Parameter((2, 1))
    lambd = cp.Parameter(nonneg=True)
    objective = cp.norm(F @ x - g) + lambd * cp.norm(x)
    constraints = [x >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    self.assertTrue(objective.is_dpp())
    self.assertTrue(constraints[0].is_dpp())
    self.assertTrue(problem.is_dpp())