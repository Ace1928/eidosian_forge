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
def test_dual_variables(self) -> None:
    """Test recovery of dual variables.
        """
    for solver in [s.ECOS, s.SCS, s.CVXOPT]:
        if solver in INSTALLED_SOLVERS:
            if solver == s.SCS:
                acc = 1
            else:
                acc = 5
            p = Problem(cp.Minimize(cp.norm1(self.x + self.z)), [self.x >= [2, 3], [[1, 2], [3, 4]] @ self.z == [-1, -4], cp.pnorm(self.x + self.z, p=2) <= 100])
            result = p.solve(solver=solver)
            self.assertAlmostEqual(result, 4, places=acc)
            self.assertItemsAlmostEqual(self.x.value, [4, 3], places=acc)
            self.assertItemsAlmostEqual(self.z.value, [-4, 1], places=acc)
            self.assertItemsAlmostEqual(p.constraints[0].dual_value, [0, 1], places=acc)
            self.assertItemsAlmostEqual(p.constraints[1].dual_value, [-1, 0.5], places=acc)
            self.assertAlmostEqual(p.constraints[2].dual_value, 0, places=acc)
            T = numpy.ones((2, 3)) * 2
            p = Problem(cp.Minimize(1), [self.A >= T @ self.C, self.A == self.B, self.C == T.T])
            result = p.solve(solver=solver)
            self.assertItemsAlmostEqual(p.constraints[0].dual_value, 4 * [0], places=acc)
            self.assertItemsAlmostEqual(p.constraints[1].dual_value, 4 * [0], places=acc)
            self.assertItemsAlmostEqual(p.constraints[2].dual_value, 6 * [0], places=acc)