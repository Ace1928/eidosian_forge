import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_affine_prod(self) -> None:
    x = Variable((3, 5))
    y = Variable((5, 4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = x @ y
    self.assertFalse(s.is_constant())
    self.assertFalse(s.is_affine())
    self.assertTrue(s.is_quadratic())
    self.assertFalse(s.is_dcp())