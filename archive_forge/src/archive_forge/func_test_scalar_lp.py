import builtins
import pickle
import sys
import warnings
from fractions import Fraction
from io import StringIO
import ecos
import numpy
import numpy as np
import scipy.sparse as sp
import scs
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, ExpCone, NonNeg, Zero
from cvxpy.error import DCPError, ParameterError, SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers import ecos_conif, scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import (
from cvxpy.reductions.solvers.solving_chain import ECOS_DEPRECATION_MSG
from cvxpy.tests.base_test import BaseTest
def test_scalar_lp(self) -> None:
    p = Problem(cp.Minimize(3 * self.a), [self.a >= 2])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 6)
    self.assertAlmostEqual(self.a.value, 2)
    p = Problem(cp.Maximize(3 * self.a - self.b), [self.a <= 2, self.b == self.a, self.b <= 5])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 4.0)
    self.assertAlmostEqual(self.a.value, 2)
    self.assertAlmostEqual(self.b.value, 2)
    p = Problem(cp.Minimize(3 * self.a - self.b + 100), [self.a >= 2, self.b + 5 * self.c - 2 == self.a, self.b <= 5 + self.c])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 101 + 1.0 / 6)
    self.assertAlmostEqual(self.a.value, 2)
    self.assertAlmostEqual(self.b.value, 5 - 1.0 / 6)
    self.assertAlmostEqual(self.c.value, -1.0 / 6)
    exp = cp.Maximize(self.a)
    p = Problem(exp, [self.a <= 2])
    result = p.solve(solver=s.ECOS)
    self.assertEqual(result, p.value)
    self.assertEqual(p.status, s.OPTIMAL)
    assert self.a.value is not None
    assert p.constraints[0].dual_value is not None
    p = Problem(cp.Maximize(self.a), [self.a >= 2])
    p.solve(solver=s.ECOS)
    self.assertEqual(p.status, s.UNBOUNDED)
    assert numpy.isinf(p.value)
    assert p.value > 0
    assert self.a.value is None
    assert p.constraints[0].dual_value is None
    if s.CVXOPT in INSTALLED_SOLVERS:
        p = Problem(cp.Minimize(-self.a), [self.a >= 2])
        result = p.solve(solver=s.CVXOPT)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.UNBOUNDED)
        assert numpy.isinf(p.value)
        assert p.value < 0
    p = Problem(cp.Maximize(self.a), [self.a >= 2, self.a <= 1])
    self.a.save_value(2)
    p.constraints[0].save_dual_value(2)
    result = p.solve(solver=s.ECOS)
    self.assertEqual(result, p.value)
    self.assertEqual(p.status, s.INFEASIBLE)
    assert numpy.isinf(p.value)
    assert p.value < 0
    assert self.a.value is None
    assert p.constraints[0].dual_value is None
    p = Problem(cp.Minimize(-self.a), [self.a >= 2, self.a <= 1])
    result = p.solve(solver=s.ECOS)
    self.assertEqual(result, p.value)
    self.assertEqual(p.status, s.INFEASIBLE)
    assert numpy.isinf(p.value)
    assert p.value > 0