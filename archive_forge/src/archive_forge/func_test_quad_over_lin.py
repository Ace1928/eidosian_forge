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
def test_quad_over_lin(self) -> None:
    atom = cp.quad_over_lin(cp.square(self.x), self.a)
    self.assertEqual(atom.curvature, s.CONVEX)
    atom = cp.quad_over_lin(-cp.square(self.x), self.a)
    self.assertEqual(atom.curvature, s.CONVEX)
    atom = cp.quad_over_lin(cp.sqrt(self.x), self.a)
    self.assertEqual(atom.curvature, s.UNKNOWN)
    assert not atom.is_dcp()
    with self.assertRaises(Exception) as cm:
        cp.quad_over_lin(self.x, self.x)
    self.assertEqual(str(cm.exception), 'The second argument to quad_over_lin must be a scalar.')