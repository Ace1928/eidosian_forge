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
def test_scalar_const_promotion(self) -> None:
    exp = self.x + 2
    self.assertEqual(exp.curvature, s.AFFINE)
    assert exp.is_affine()
    self.assertEqual(exp.sign, s.UNKNOWN)
    assert not exp.is_nonpos()
    self.assertEqual(exp.shape, (2,))
    self.assertEqual((4 - self.x).shape, (2,))
    self.assertEqual((4 * self.x).shape, (2,))
    self.assertEqual((4 <= self.x).shape, (2,))
    self.assertEqual((4 == self.x).shape, (2,))
    self.assertEqual((self.x >= 4).shape, (2,))
    exp = self.A + 2 + 4
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual((3 * self.A).shape, (2, 2))
    self.assertEqual(exp.shape, (2, 2))