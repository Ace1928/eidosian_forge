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
def test_assign_var_value(self) -> None:
    """Test assigning a value to a variable.
        """
    a = Variable()
    a.value = 1
    self.assertEqual(a.value, 1)
    with self.assertRaises(Exception) as cm:
        a.value = [2, 1]
    self.assertEqual(str(cm.exception), 'Invalid dimensions (2,) for Variable value.')
    a.value = 1
    a.value = None
    assert a.value is None
    x = Variable(2)
    x.value = [2, 1]
    self.assertItemsAlmostEqual(x.value, [2, 1])
    A = Variable((3, 2))
    A.value = np.ones((3, 2))
    self.assertItemsAlmostEqual(A.value, np.ones((3, 2)))
    x = Variable(nonneg=True)
    with self.assertRaises(Exception) as cm:
        x.value = -2
    self.assertEqual(str(cm.exception), 'Variable value must be nonnegative.')