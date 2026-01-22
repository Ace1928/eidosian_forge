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
def test_var_copy(self) -> None:
    """Test the copy function for variable types.
        """
    x = Variable((3, 4), name='x')
    y = x.copy()
    self.assertEqual(y.shape, (3, 4))
    self.assertEqual(y.name(), 'x')
    x = Variable((5, 5), PSD=True, name='x')
    y = x.copy()
    self.assertEqual(y.shape, (5, 5))