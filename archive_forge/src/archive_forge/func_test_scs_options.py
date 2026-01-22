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
def test_scs_options(self) -> None:
    """Test that all the SCS solver options work.
        """
    EPS = 0.0001
    x = cp.Variable(2, name='x')
    prob = cp.Problem(cp.Minimize(cp.norm(x, 1) + 1.0), [x == 0])
    for i in range(2):
        prob.solve(solver=cp.SCS, max_iters=50, eps=EPS, alpha=1.2, verbose=True, normalize=True, use_indirect=False)
    self.assertAlmostEqual(prob.value, 1.0, places=2)
    self.assertItemsAlmostEqual(x.value, [0, 0], places=2)