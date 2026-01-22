import unittest
from traits.api import HasTraits, Int, List
def test_extend_via_inplace_addition(self):
    foo = MyClass()
    foo.l += [4, 5, 6]
    self.assertEqual(foo.l, [1, 2, 3, 4, 5, 6])
    self.assertEqual(len(foo.l_events), 1)
    event = foo.l_events[0]
    self.assertEqual(event.added, [4, 5, 6])
    self.assertEqual(event.removed, [])
    self.assertEqual(event.index, 3)