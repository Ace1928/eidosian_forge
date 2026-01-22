import unittest
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def int_prob(self, solver) -> None:
    obj = cp.Minimize(cp.abs(self.y_int - 0.2))
    p = cp.Problem(obj, [])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0.2)
    self.assertAlmostEqual(self.y_int.value, 0)
    t = cp.Variable()
    obj = cp.Minimize(t)
    p = cp.Problem(obj, [self.y_int == 0.5, t >= 0])
    result = p.solve(solver=solver)
    self.assertEqual(p.status in s.INF_OR_UNB, True)