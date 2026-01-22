import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_ignore_dpp(self) -> None:
    """Test the ignore_dpp flag.
        """
    x = cp.Parameter()
    x.value = 5
    y = cp.Variable()
    problem = cp.Problem(cp.Minimize(x + y), [x == y])
    self.assertTrue(problem.is_dpp())
    self.assertTrue(problem.is_dcp())
    result = problem.solve(cp.SCS, ignore_dpp=True)
    self.assertAlmostEqual(result, 10)
    with pytest.raises(error.DPPError):
        problem.solve(cp.SCS, enforce_dpp=True, ignore_dpp=True)