import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_mul_2d(self) -> None:
    """Test multiplication where at least one of the shapes is >= 2D.
        """
    self.assertEqual(shape.mul_shapes((5, 9), (9, 2)), (5, 2))
    self.assertEqual(shape.mul_shapes((3, 5, 9), (3, 9, 2)), (3, 5, 2))
    with self.assertRaises(Exception) as cm:
        shape.mul_shapes((5, 3), (9, 2))
    self.assertEqual(str(cm.exception), 'Incompatible dimensions (5, 3) (9, 2)')
    with self.assertRaises(Exception) as cm:
        shape.mul_shapes((3, 5, 9), (4, 9, 2))
    self.assertEqual(str(cm.exception), 'Incompatible dimensions (3, 5, 9) (4, 9, 2)')