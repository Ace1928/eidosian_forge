import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_log_log_curvature(self) -> None:
    """Test that the curvature string is populated for log-log expressions.
        """
    x = Variable(pos=True)
    monomial = x * x * x
    assert monomial.curvature == s.LOG_LOG_AFFINE
    posynomial = x * x * x + x
    assert posynomial.curvature == s.LOG_LOG_CONVEX
    llcv = 1 / (x * x * x + x)
    assert llcv.curvature == s.LOG_LOG_CONCAVE