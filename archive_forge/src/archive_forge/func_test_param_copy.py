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
def test_param_copy(self) -> None:
    """Test the copy function for Parameters.
        """
    x = Parameter((3, 4), name='x', nonneg=True)
    y = x.copy()
    self.assertEqual(y.shape, (3, 4))
    self.assertEqual(y.name(), 'x')
    self.assertEqual(y.sign, 'NONNEGATIVE')