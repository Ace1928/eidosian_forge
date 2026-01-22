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
def test_ecos_options(self) -> None:
    """Test that all the ECOS solver options work.
        """
    EPS = 0.0001
    prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
    for i in range(2):
        prob.solve(solver=cp.ECOS, feastol=EPS, abstol=EPS, reltol=EPS, feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS, max_iters=20, verbose=True, warm_start=True)
    self.assertAlmostEqual(prob.value, 1.0)
    self.assertItemsAlmostEqual(self.x.value, [0, 0])