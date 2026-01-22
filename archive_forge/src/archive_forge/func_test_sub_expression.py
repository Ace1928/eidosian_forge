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
def test_sub_expression(self) -> None:
    c = Constant([2, 2])
    exp = self.x - c
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.sign, s.UNKNOWN)
    self.assertEqual(exp.shape, (2,))
    z = Variable(2, name='z')
    exp = exp - z - self.x
    with self.assertRaises(ValueError):
        self.x - self.y
    exp = self.A - self.B
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (2, 2))
    with self.assertRaises(ValueError):
        self.A - self.C
    self.assertEqual(repr(self.x - c), 'Expression(AFFINE, UNKNOWN, (2,))')