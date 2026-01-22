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
def test_cp_node_count_warn(self) -> None:
    """Test that a warning is raised for high node count."""
    with warnings.catch_warnings(record=True) as w:
        a = cp.Variable(shape=(100, 100))
        b = sum((sum(x) for x in a))
        cp.Problem(cp.Maximize(0), [b >= 0])
        assert len(w) == 1
        assert 'vectorizing' in str(w[-1].message)
        assert 'Constraint #0' in str(w[-1].message)
    with warnings.catch_warnings(record=True) as w:
        a = cp.Variable(shape=(100, 100))
        b = sum((sum(x) for x in a))
        cp.Problem(cp.Maximize(b))
        assert len(w) == 1
        assert 'vectorizing' in str(w[-1].message)
        assert 'Objective' in str(w[-1].message)
    with warnings.catch_warnings(record=True) as w:
        a = cp.Variable(shape=(100, 100))
        c = cp.sum(a)
        cp.Problem(cp.Maximize(0), [c >= 0])
        assert len(w) == 0