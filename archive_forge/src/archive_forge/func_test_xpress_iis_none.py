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
def test_xpress_iis_none(self) -> None:
    if cp.XPRESS in INSTALLED_SOLVERS:
        A = np.array([[2, 1], [1, 2], [-3, -3]])
        b = np.array([2, 2, -5])
        x = cp.Variable(2)
        objective = cp.Minimize(cp.norm2(x))
        constraint = [A @ x <= b]
        problem = cp.Problem(objective, constraint)
        params = {'save_iis': 0}
        problem.solve(solver=cp.XPRESS, **params)