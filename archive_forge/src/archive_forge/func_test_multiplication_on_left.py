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
def test_multiplication_on_left(self) -> None:
    """Test multiplication on the left by a non-constant.
        """
    c = numpy.array([[1, 2]]).T
    p = Problem(cp.Minimize(c.T @ self.A @ c), [self.A >= 2])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 18)
    p = Problem(cp.Minimize(self.a * 2), [self.a >= 2])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 4)
    p = Problem(cp.Minimize(self.x.T @ c), [self.x >= 2])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 6)
    p = Problem(cp.Minimize((self.x.T + self.z.T) @ c), [self.x >= 2, self.z >= 1])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 9)
    A = numpy.ones((5, 10))
    x = Variable(5)
    p = cp.Problem(cp.Minimize(cp.sum(x @ A)), [x >= 0])
    result = p.solve(solver=cp.SCS, eps=1e-06)
    self.assertAlmostEqual(result, 0)