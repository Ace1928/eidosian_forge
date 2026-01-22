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
def test_invalid_solvers(self) -> None:
    """Tests that errors occur when you use an invalid solver.
        """
    with self.assertRaises(SolverError):
        Problem(cp.Minimize(Variable(boolean=True))).solve(solver=s.ECOS)
    with self.assertRaises(SolverError):
        Problem(cp.Minimize(cp.lambda_max(self.A))).solve(solver=s.ECOS)
    with self.assertRaises(SolverError):
        Problem(cp.Minimize(self.a)).solve(solver=s.SCS)