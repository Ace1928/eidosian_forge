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
def test_lin_combination_problems(self) -> None:
    prob1 = Problem(cp.Minimize(self.a), [self.a >= self.b])
    prob2 = Problem(cp.Minimize(2 * self.b), [self.a >= 1, self.b >= 2])
    prob3 = Problem(cp.Maximize(-pow(self.b + self.a, 2)), [self.b >= 3])
    combo1 = prob1 + 2 * prob2
    combo1_ref = Problem(cp.Minimize(self.a + 4 * self.b), [self.a >= self.b, self.a >= 1, self.b >= 2])
    self.assertAlmostEqual(combo1.solve(solver=cp.ECOS), combo1_ref.solve(solver=cp.ECOS))
    combo2 = prob1 - prob3 / 2
    combo2_ref = Problem(cp.Minimize(self.a + pow(self.b + self.a, 2) / 2), [self.b >= 3, self.a >= self.b])
    self.assertAlmostEqual(combo2.solve(solver=cp.ECOS), combo2_ref.solve(solver=cp.ECOS))
    combo3 = prob1 + 0 * prob2 - 3 * prob3
    combo3_ref = Problem(cp.Minimize(self.a + 3 * pow(self.b + self.a, 2)), [self.a >= self.b, self.a >= 1, self.b >= 3])
    self.assertAlmostEqual(combo3.solve(solver=cp.ECOS), combo3_ref.solve(solver=cp.ECOS))