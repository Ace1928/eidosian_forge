import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_add_matching(self) -> None:
    """Test addition of matching shapes.
        """
    self.assertEqual(shape.sum_shapes([(3, 4), (3, 4)]), (3, 4))
    self.assertEqual(shape.sum_shapes([(3, 4)] * 5), (3, 4))