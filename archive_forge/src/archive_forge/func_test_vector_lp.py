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
def test_vector_lp(self) -> None:
    c = Constant(numpy.array([[1, 2]]).T).value
    p = Problem(cp.Minimize(c.T @ self.x), [self.x[:, None] >= c])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 5)
    self.assertItemsAlmostEqual(self.x.value, [1, 2])
    A = Constant(numpy.array([[3, 5], [1, 2]]).T).value
    Imat = Constant([[1, 0], [0, 1]])
    p = Problem(cp.Minimize(c.T @ self.x + self.a), [A @ self.x >= [-1, 1], 4 * Imat @ self.z == self.x, self.z >= [2, 2], self.a >= 2])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 26, places=3)
    obj = (c.T @ self.x + self.a).value[0]
    self.assertAlmostEqual(obj, result)
    self.assertItemsAlmostEqual(self.x.value, [8, 8], places=3)
    self.assertItemsAlmostEqual(self.z.value, [2, 2], places=3)