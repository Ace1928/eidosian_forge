import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_non_quadratic(self) -> None:
    x = Variable()
    y = Variable()
    z = Variable()
    s = cp.max(vstack([x, y, z])) ** 2
    self.assertFalse(s.is_quadratic())
    t = cp.max(vstack([x ** 2, power(y, 2), z]))
    self.assertFalse(t.is_quadratic())