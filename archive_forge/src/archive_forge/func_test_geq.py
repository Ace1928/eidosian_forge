import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_geq(self) -> None:
    """Test the >= operator.
        """
    constr = self.z >= self.x
    self.assertEqual(constr.name(), 'x <= z')
    self.assertEqual(constr.shape, (2,))
    with self.assertRaises(ValueError):
        self.y >= self.x