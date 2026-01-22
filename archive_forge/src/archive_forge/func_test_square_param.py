import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
import cvxpy as cp
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import (
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs
def test_square_param(self) -> None:
    """Test issue arising with square plus parameter.
        """
    a = Parameter(value=1)
    b = Variable()
    obj = Minimize(b ** 2 + abs(a))
    prob = Problem(obj)
    prob.solve(solver='SCS')
    self.assertAlmostEqual(obj.value, 1.0)