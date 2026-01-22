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
def test_sigma_max(self) -> None:
    """Test sigma_max.
        """
    const = cp.Constant([[1, 2, 3], [4, 5, 6]])
    constr = [self.C == const]
    prob = cp.Problem(cp.Minimize(cp.norm(self.C, 2)), constr)
    result = prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, cp.norm(const, 2).value)
    self.assertItemsAlmostEqual(self.C.value, const.value)