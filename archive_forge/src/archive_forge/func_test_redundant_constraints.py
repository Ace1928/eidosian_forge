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
def test_redundant_constraints(self) -> None:
    obj = cp.Minimize(cp.sum(self.x))
    constraints = [self.x == 2, self.x == 2, self.x.T == 2, self.x[0] == 2]
    p = Problem(obj, constraints)
    result = p.solve(solver=s.SCS)
    self.assertAlmostEqual(result, 4)
    obj = cp.Minimize(cp.sum(cp.square(self.x)))
    constraints = [self.x == self.x]
    p = Problem(obj, constraints)
    result = p.solve(solver=s.SCS)
    self.assertAlmostEqual(result, 0)
    with self.assertRaises(ValueError) as cm:
        obj = cp.Minimize(cp.sum(cp.square(self.x)))
        constraints = [self.x == self.x]
        problem = Problem(obj, constraints)
        problem.solve(solver=s.ECOS)
    self.assertEqual(str(cm.exception), 'ECOS cannot handle sparse data with nnz == 0; this is a bug in ECOS, and it indicates that your problem might have redundant constraints.')