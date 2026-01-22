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
def test_multiply_by_scalar(self) -> None:
    """Test a problem with multiply by a scalar.
        """
    import numpy as np
    T = 10
    J = 20
    rvec = np.random.randn(T, J)
    dy = np.random.randn(2 * T)
    theta = Variable(J)
    delta = 0.001
    loglambda = rvec @ theta
    a = cp.multiply(dy[0:T], loglambda)
    b1 = cp.exp(loglambda)
    b2 = cp.multiply(delta, b1)
    cost = -a + b1
    cost = -a + b2
    prob = Problem(cp.Minimize(cp.sum(cost)))
    prob.solve(solver=s.SCS)
    obj = cp.Minimize(cp.sum(cp.multiply(2, self.x)))
    prob = Problem(obj, [self.x == 2])
    result = prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, 8)