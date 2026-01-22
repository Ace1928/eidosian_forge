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
def test_is_qp(self) -> None:
    A = numpy.random.randn(4, 3)
    b = numpy.random.randn(4)
    Aeq = numpy.random.randn(2, 3)
    beq = numpy.random.randn(2)
    F = numpy.random.randn(2, 3)
    g = numpy.random.randn(2)
    obj = cp.sum_squares(A @ self.y - b)
    qpwa_obj = 3 * cp.sum_squares(-cp.abs(A @ self.y)) + cp.quad_over_lin(cp.maximum(cp.abs(A @ self.y), [3.0, 3.0, 3.0, 3.0]), 2.0)
    not_qpwa_obj = 3 * cp.sum_squares(cp.abs(A @ self.y)) + cp.quad_over_lin(cp.minimum(cp.abs(A @ self.y), [3.0, 3.0, 3.0, 3.0]), 2.0)
    p = Problem(cp.Minimize(obj), [])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(qpwa_obj), [])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(not_qpwa_obj), [])
    self.assertEqual(p.is_qp(), False)
    p = Problem(cp.Minimize(obj), [Aeq @ self.y == beq, F @ self.y <= g])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(qpwa_obj), [Aeq @ self.y == beq, F @ self.y <= g])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y) <= 200, cp.abs(2 * self.y) <= 100, cp.norm(2 * self.y, 1) <= 1000, Aeq @ self.y == beq])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y) <= 200, cp.abs(2 * self.y) <= 100, cp.norm(2 * self.y, 1) <= 1000, Aeq @ self.y == beq])
    self.assertEqual(p.is_qp(), True)
    p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
    self.assertEqual(p.is_qp(), False)
    p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
    self.assertEqual(p.is_qp(), False)