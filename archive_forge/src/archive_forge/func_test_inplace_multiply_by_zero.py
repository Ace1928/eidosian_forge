import unittest
from traits.api import HasTraits, Int, List
def test_inplace_multiply_by_zero(self):
    foo = MyClass()
    foo.l *= 0
    self.assertEqual(foo.l, [])
    self.assertEqual(len(foo.l_events), 1)
    event = foo.l_events[0]
    self.assertEqual(event.added, [])
    self.assertEqual(event.removed, [1, 2, 3])
    self.assertEqual(event.index, 0)