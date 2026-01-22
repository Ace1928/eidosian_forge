import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_can_solve_non_dpp_problem(self) -> None:
    x = cp.Parameter()
    x.value = 5
    y = cp.Variable()
    problem = cp.Problem(cp.Minimize(x * x), [x == y])
    self.assertFalse(problem.is_dpp())
    self.assertTrue(problem.is_dcp())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        self.assertEqual(problem.solve(cp.SCS), 25)
    x.value = 3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        self.assertEqual(problem.solve(cp.SCS), 9)