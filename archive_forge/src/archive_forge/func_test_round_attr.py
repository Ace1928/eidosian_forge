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
def test_round_attr(self) -> None:
    """Test rounding for attributes.
        """
    v = Variable(1, nonpos=True)
    self.assertAlmostEqual(v.project(1), 0)
    v = Variable(2, nonpos=True)
    self.assertItemsAlmostEqual(v.project(np.array([1, -1])), [0, -1])
    v = Variable(1, nonneg=True)
    self.assertAlmostEqual(v.project(-1), 0)
    v = Variable(2, nonneg=True)
    self.assertItemsAlmostEqual(v.project(np.array([1, -1])), [1, 0])
    v = Variable((2, 2), boolean=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]]).T), [1, 0, 1, 0])
    v = Variable((2, 2), integer=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1.6], [1, 0]]).T), [1, -2, 1, 0])
    v = Variable((2, 2), symmetric=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]])), [1, 0, 0, 0])
    v = Variable((2, 2), PSD=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, -1]])), [1, 0, 0, 0])
    v = Variable((2, 2), NSD=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, -1]])), [0, 0, 0, -1])
    v = Variable((2, 2), diag=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]])).todense(), [1, 0, 0, 0])
    v = Variable((2, 2), hermitian=True)
    self.assertItemsAlmostEqual(v.project(np.array([[1, -1j], [1, 0]])), [1, 0.5 + 0.5j, 0.5 - 0.5j, 0])
    A = Constant(np.array([[1.0]]))
    self.assertEqual(A.is_psd(), True)
    self.assertEqual(A.is_nsd(), False)
    A = Constant(np.array([[-1.0]]))
    self.assertEqual(A.is_psd(), False)
    self.assertEqual(A.is_nsd(), True)
    A = Constant(np.array([[0.0]]))
    self.assertEqual(A.is_psd(), True)
    self.assertEqual(A.is_nsd(), True)