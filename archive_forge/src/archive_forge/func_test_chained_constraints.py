import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_chained_constraints(self) -> None:
    """Tests that chaining constraints raises an error.
        """
    error_str = 'Cannot evaluate the truth value of a constraint or chain constraints, e.g., 1 >= x >= 0.'
    with self.assertRaises(Exception) as cm:
        self.z <= self.x <= 1
    self.assertEqual(str(cm.exception), error_str)
    with self.assertRaises(Exception) as cm:
        self.x == self.z == 1
    self.assertEqual(str(cm.exception), error_str)
    with self.assertRaises(Exception) as cm:
        (self.z <= self.x).__bool__()
    self.assertEqual(str(cm.exception), error_str)