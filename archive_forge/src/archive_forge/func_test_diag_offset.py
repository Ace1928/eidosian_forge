import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_diag_offset(self) -> None:
    """Test matrix to vector on scalar matrices"""
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_vector = np.array([1, 2, 3])
    offsets = [0, 1, -1, 2]
    for offset in offsets:
        a_cp = cp.diag(test_matrix, k=offset)
        a_np = np.diag(test_matrix, k=offset)
        A_cp = cp.diag(test_vector, k=offset)
        A_np = np.diag(test_vector, k=offset)
        self.assertItemsAlmostEqual(a_cp.value, a_np)
        self.assertItemsAlmostEqual(A_cp.value, A_np)
    X = cp.diag(Variable(5), 1)
    self.assertEqual(X.size, 36)