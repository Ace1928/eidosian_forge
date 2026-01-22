import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_nonlla_equality_constraint_not_dpp(self) -> None:
    alpha = cp.Parameter(pos=True, value=1.0)
    x = cp.Variable(pos=True)
    constraint = [x == x + alpha]
    self.assertFalse(constraint[0].is_dgp(dpp=True))
    self.assertFalse(cp.Problem(cp.Minimize(1), constraint).is_dgp(dpp=True))