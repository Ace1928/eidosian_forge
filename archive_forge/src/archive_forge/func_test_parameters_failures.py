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
def test_parameters_failures(self) -> None:
    p = Parameter(name='p')
    self.assertEqual(p.name(), 'p')
    self.assertEqual(p.shape, tuple())
    p = Parameter((4, 3), nonneg=True)
    with self.assertRaises(Exception) as cm:
        p.value = 1
    self.assertEqual(str(cm.exception), 'Invalid dimensions () for Parameter value.')
    val = -np.ones((4, 3))
    val[0, 0] = 2
    p = Parameter((4, 3), nonneg=True)
    with self.assertRaises(Exception) as cm:
        p.value = val
    self.assertEqual(str(cm.exception), 'Parameter value must be nonnegative.')
    p = Parameter((4, 3), nonpos=True)
    with self.assertRaises(Exception) as cm:
        p.value = val
    self.assertEqual(str(cm.exception), 'Parameter value must be nonpositive.')
    with self.assertRaises(Exception) as cm:
        p = Parameter(2, 1, nonpos=True, value=[2, 1])
    self.assertEqual(str(cm.exception), 'Parameter value must be nonpositive.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((4, 3), nonneg=True, value=[1, 2])
    self.assertEqual(str(cm.exception), 'Invalid dimensions (2,) for Parameter value.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((2, 2), diag=True, symmetric=True)
    self.assertEqual(str(cm.exception), 'Cannot set more than one special attribute in Parameter.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((2, 2), boolean=True, value=[[1, 1], [1, -1]])
    self.assertEqual(str(cm.exception), 'Parameter value must be boolean.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((2, 2), integer=True, value=[[1, 1.5], [1, -1]])
    self.assertEqual(str(cm.exception), 'Parameter value must be integer.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((2, 2), diag=True, value=[[1, 1], [1, -1]])
    self.assertEqual(str(cm.exception), 'Parameter value must be diagonal.')
    with self.assertRaises(Exception) as cm:
        p = Parameter((2, 2), symmetric=True, value=[[1, 1], [-1, -1]])
    self.assertEqual(str(cm.exception), 'Parameter value must be symmetric.')