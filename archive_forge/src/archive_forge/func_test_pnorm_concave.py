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
def test_pnorm_concave(self) -> None:
    import numpy as np
    x = Variable(3, name='x')
    a = np.array([-1.0, 2, 3])
    for p in (-1, 0.5, 0.3, -2.3):
        prob = Problem(cp.Minimize(cp.sum(cp.abs(x - a))), [cp.pnorm(x, p) >= 0])
        prob.solve(solver=cp.ECOS)
        self.assertTrue(np.allclose(prob.value, 1))
    a = np.array([1.0, 2, 3])
    for p in (-1, 0.5, 0.3, -2.3):
        prob = Problem(cp.Minimize(cp.sum(cp.abs(x - a))), [cp.pnorm(x, p) >= 0])
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(prob.value, 0, places=6)