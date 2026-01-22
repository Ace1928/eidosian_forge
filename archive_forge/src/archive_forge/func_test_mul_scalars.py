import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_mul_scalars(self) -> None:
    """Test multiplication by scalars raises a ValueError.
        """
    with self.assertRaises(ValueError):
        shape.mul_shapes(tuple(), (5, 9))
    with self.assertRaises(ValueError):
        shape.mul_shapes((5, 9), tuple())
    with self.assertRaises(ValueError):
        shape.mul_shapes(tuple(), tuple())