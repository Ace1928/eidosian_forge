import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_add_broadcasting(self) -> None:
    """Test broadcasting of shapes during addition.
        """
    self.assertEqual(shape.sum_shapes([(3, 4), (1, 1)]), (3, 4))
    self.assertEqual(shape.sum_shapes([(1, 1), (3, 4)]), (3, 4))
    self.assertEqual(shape.sum_shapes([(1,), (3, 4)]), (3, 4))
    self.assertEqual(shape.sum_shapes([(3, 4), (1,)]), (3, 4))
    self.assertEqual(shape.sum_shapes([tuple(), (3, 4)]), (3, 4))
    self.assertEqual(shape.sum_shapes([(3, 4), tuple()]), (3, 4))
    self.assertEqual(shape.sum_shapes([(1, 1), (4,)]), (1, 4))
    self.assertEqual(shape.sum_shapes([(4,), (1, 1)]), (1, 4))
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4, 1), (4,)])
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4,), (4, 1)])
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4, 2), (2,)])
    with self.assertRaises(ValueError):
        shape.sum_shapes([(2,), (4, 2)])
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4, 2), (4, 1)])
    with self.assertRaises(ValueError):
        shape.sum_shapes([(4, 1), (4, 2)])