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
def test_partial_optimize_numeric_fn(self) -> None:
    x, y = (Variable(), Variable())
    xval = 4
    p1 = Problem(Minimize(y), [xval + y >= 3])
    p1.solve(solver=cp.SCS, eps=1e-06)
    constr = [y >= -100]
    p2 = Problem(Minimize(y), [x + y >= 3] + constr)
    g = partial_optimize(p2, [y], [x], solver=cp.SCS, eps=1e-06)
    x.value = xval
    y.value = 42
    constr[0].dual_variables[0].value = 42
    result = g.value
    self.assertAlmostEqual(result, p1.value)
    self.assertAlmostEqual(y.value, 42)
    self.assertAlmostEqual(constr[0].dual_value, 42)
    p2 = Problem(Minimize(y), [x + y >= 3])
    g = partial_optimize(p2, [], [x, y], solver=cp.SCS, eps=1e-06)
    x.value = xval
    y.value = 42
    p2.constraints[0].dual_variables[0].value = 42
    result = g.value
    self.assertAlmostEqual(result, y.value)
    self.assertAlmostEqual(y.value, 42)
    self.assertAlmostEqual(p2.constraints[0].dual_value, 42)