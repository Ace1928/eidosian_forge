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
def test_norm_inf(self) -> None:
    """Test the norm_inf class.
        """
    exp = self.x + self.y
    atom = cp.norm_inf(exp)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    assert atom.is_convex()
    assert (-atom).is_concave()
    self.assertEqual(cp.norm_inf(atom).curvature, s.CONVEX)
    self.assertEqual(cp.norm_inf(-atom).curvature, s.CONVEX)