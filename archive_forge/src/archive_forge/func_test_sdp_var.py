import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_sdp_var(self) -> None:
    """Test sdp var.
        """
    const = cp.Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X = cp.Variable((3, 3), PSD=True)
    prob = cp.Problem(cp.Minimize(0), [X == const])
    prob.solve(solver=cp.SCS)
    self.assertEqual(prob.status, cp.INFEASIBLE)