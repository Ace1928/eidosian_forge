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
def test_presolve_parameters(self) -> None:
    """Test presolve with parameters.
        """
    gamma = Parameter(nonneg=True)
    x = Variable()
    obj = cp.Minimize(x)
    prob = Problem(obj, [gamma == 1, x >= 0])
    gamma.value = 0
    prob.solve(solver=s.SCS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    gamma.value = 1
    prob.solve(solver=s.SCS)
    self.assertEqual(prob.status, s.OPTIMAL)