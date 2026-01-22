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
def test_min_with_axis(self) -> None:
    """Test reshape of a min with axis=0.
        """
    x = cp.Variable((5, 2))
    y = cp.Variable((5, 2))
    stacked_flattened = cp.vstack([cp.vec(x), cp.vec(y)])
    minimum = cp.min(stacked_flattened, axis=0)
    reshaped_minimum = cp.reshape(minimum, (5, 2))
    obj = cp.sum(reshaped_minimum)
    problem = cp.Problem(cp.Maximize(obj), [x == 1, y == 2])
    result = problem.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, 10)