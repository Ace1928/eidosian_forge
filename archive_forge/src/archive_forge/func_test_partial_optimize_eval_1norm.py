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
def test_partial_optimize_eval_1norm(self) -> None:
    """Test the partial_optimize atom.
        """
    dims = 3
    x, t = (Variable(dims), Variable(dims))
    xval = [-5] * dims
    p1 = Problem(cp.Minimize(cp.sum(t)), [-t <= xval, xval <= t])
    p1.solve(solver='ECOS')
    p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
    g = partial_optimize(p2, [t], [x], solver='ECOS')
    p3 = Problem(cp.Minimize(g), [x == xval])
    p3.solve(solver='ECOS')
    self.assertAlmostEqual(p1.value, p3.value)
    p2 = Problem(cp.Maximize(cp.sum(-t)), [-t <= x, x <= t])
    g = partial_optimize(p2, opt_vars=[t], solver='ECOS')
    p3 = Problem(cp.Maximize(g), [x == xval])
    p3.solve(solver='ECOS')
    self.assertAlmostEqual(p1.value, -p3.value)
    p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
    g = partial_optimize(p2, opt_vars=[t], solver='ECOS')
    p3 = Problem(cp.Minimize(g), [x == xval])
    p3.solve(solver='ECOS')
    self.assertAlmostEqual(p1.value, p3.value)
    g = partial_optimize(p2, dont_opt_vars=[x], solver='ECOS')
    p3 = Problem(cp.Minimize(g), [x == xval])
    p3.solve(solver='ECOS')
    self.assertAlmostEqual(p1.value, p3.value)
    with self.assertRaises(Exception) as cm:
        g = partial_optimize(p2, solver='ECOS')
    self.assertEqual(str(cm.exception), 'partial_optimize called with neither opt_vars nor dont_opt_vars.')
    with self.assertRaises(Exception) as cm:
        g = partial_optimize(p2, [], [x], solver='ECOS')
    self.assertEqual(str(cm.exception), 'If opt_vars and new_opt_vars are both specified, they must contain all variables in the problem.')