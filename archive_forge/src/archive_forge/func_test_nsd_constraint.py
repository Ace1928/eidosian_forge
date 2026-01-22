import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_nsd_constraint(self) -> None:
    """Test the PSD constraint <<.
        """
    constr = self.A << self.B
    self.assertEqual(constr.name(), 'B + -A >> 0')
    self.assertEqual(constr.shape, (2, 2))
    assert constr.dual_value is None
    with self.assertRaises(ValueError):
        constr.value()
    self.B.save_value(np.array([[2, -1], [1, 2]]))
    self.A.save_value(np.array([[1, 0], [0, 1]]))
    assert constr.value()
    self.A.save_value(np.array([[3, 0], [0, 3]]))
    assert not constr.value()
    with self.assertRaises(Exception) as cm:
        self.x << 0
    self.assertEqual(str(cm.exception), 'Non-square matrix in positive definite constraint.')