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
def test_transpose_variable(self) -> None:
    var = self.a.T
    self.assertEqual(var.name(), 'a')
    self.assertEqual(var.shape, tuple())
    self.a.save_value(2)
    self.assertEqual(var.value, 2)
    var = self.x
    self.assertEqual(var.name(), 'x')
    self.assertEqual(var.shape, (2,))
    x = Variable((2, 1), name='x')
    var = x.T
    self.assertEqual(var.name(), 'x.T')
    self.assertEqual(var.shape, (1, 2))
    x.save_value(np.array([[1, 2]]).T)
    self.assertEqual(var.value[0, 0], 1)
    self.assertEqual(var.value[0, 1], 2)
    var = self.C.T
    self.assertEqual(var.name(), 'C.T')
    self.assertEqual(var.shape, (2, 3))
    index = var[1, 0]
    self.assertEqual(index.name(), 'C.T[1, 0]')
    self.assertEqual(index.shape, tuple())
    var = x.T.T
    self.assertEqual(var.name(), 'x.T.T')
    self.assertEqual(var.shape, (2, 1))