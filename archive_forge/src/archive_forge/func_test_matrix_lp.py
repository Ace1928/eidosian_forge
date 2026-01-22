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
def test_matrix_lp(self) -> None:
    T = Constant(numpy.ones((2, 2))).value
    p = Problem(cp.Minimize(1), [self.A == T])
    result = p.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, 1)
    self.assertItemsAlmostEqual(self.A.value, T)
    T = Constant(numpy.ones((2, 3)) * 2).value
    p = Problem(cp.Minimize(1), [self.A >= T @ self.C, self.A == self.B, self.C == T.T])
    result = p.solve(solver=cp.ECOS)
    self.assertAlmostEqual(result, 1)
    self.assertItemsAlmostEqual(self.A.value, self.B.value)
    self.assertItemsAlmostEqual(self.C.value, T)
    assert (self.A.value >= (T @ self.C).value).all()
    self.assertEqual(type(self.A.value), intf.DEFAULT_INTF.TARGET_MATRIX)