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
def test_partial_transpose_exceptions(self) -> None:
    """Test exceptions raised by partial trace.
        """
    X = cp.Variable((4, 3))
    with self.assertRaises(ValueError) as cm:
        cp.partial_transpose(X, dims=[2, 3], axis=0)
    self.assertEqual(str(cm.exception), 'Only supports square matrices.')
    X = cp.Variable((6, 6))
    with self.assertRaises(ValueError) as cm:
        cp.partial_transpose(X, dims=[2, 3], axis=-1)
    self.assertEqual(str(cm.exception), 'Invalid axis argument, should be between 0 and 2, got -1.')
    X = cp.Variable((6, 6))
    with self.assertRaises(ValueError) as cm:
        cp.partial_transpose(X, dims=[2, 4], axis=0)
    self.assertEqual(str(cm.exception), "Dimension of system doesn't correspond to dimension of subsystems.")