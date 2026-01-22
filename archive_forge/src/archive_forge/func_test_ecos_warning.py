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
def test_ecos_warning(self) -> None:
    """Test that a warning is raised when ECOS
           is called by default.
        """
    x = cp.Variable()
    prob = cp.Problem(cp.Maximize(x), [x ** 2 <= 1])
    candidate_solvers = prob._find_candidate_solvers(solver=None, gp=False)
    prob._sort_candidate_solvers(candidate_solvers)
    if candidate_solvers['conic_solvers'][0] == cp.ECOS:
        with warnings.catch_warnings(record=True) as w:
            prob.solve()
            assert isinstance(w[0].message, FutureWarning)
            assert str(w[0].message) == ECOS_DEPRECATION_MSG
        with warnings.catch_warnings(record=True) as w:
            prob.solve(solver=cp.ECOS)
            assert len(w) == 0