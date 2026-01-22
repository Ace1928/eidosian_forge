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
def test_mosek_params(self) -> None:
    import mosek
    n = 10
    m = 4
    np.random.seed(0)
    A = np.random.randn(m, n)
    x = np.random.randn(n)
    y = A.dot(x)
    z = cp.Variable(n)
    objective = cp.Minimize(cp.norm1(z))
    constraints = [A @ z == y]
    problem = cp.Problem(objective, constraints)
    invalid_mosek_params = {'MSK_IPAR_NUM_THREADS': '11.3'}
    with self.assertRaises(mosek.Error):
        problem.solve(solver=cp.MOSEK, mosek_params=invalid_mosek_params)
    with self.assertRaises(ValueError):
        problem.solve(solver=cp.MOSEK, invalid_kwarg=None)
    mosek_params = {mosek.dparam.basis_tol_x: 1e-08, 'MSK_IPAR_INTPNT_MAX_ITERATIONS': 20, 'MSK_IPAR_NUM_THREADS': '17', 'MSK_IPAR_PRESOLVE_USE': 'MSK_PRESOLVE_MODE_OFF', 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-09, 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': '1e-9'}
    with pytest.warns():
        problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)