import unittest
from traits.api import Float, HasTraits, Int, List
def test_add_trait(self):
    foo = Foo(x=3)
    foo.add_trait('y', Int)
    self.assertTrue(hasattr(foo, 'y'))
    self.assertEqual(type(foo.y), int)
    foo.y = 4
    self.assertEqual(foo.y_changes, [4])