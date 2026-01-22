import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_paper_example_stoch_control(self) -> None:
    n, m = (3, 3)
    x = cp.Parameter((n, 1))
    P_sqrt = cp.Parameter((m, m))
    P_21 = cp.Parameter((n, m))
    q = cp.Parameter((m, 1))
    u = cp.Variable((m, 1))
    y = cp.Variable((n, 1))
    objective = 0.5 * cp.sum_squares(P_sqrt @ u) + x.T @ y + q.T @ u
    problem = cp.Problem(cp.Minimize(objective), [cp.norm(u) <= 0.5, y == P_21 @ u])
    self.assertTrue(problem.is_dpp())
    self.assertTrue(problem.is_dcp())