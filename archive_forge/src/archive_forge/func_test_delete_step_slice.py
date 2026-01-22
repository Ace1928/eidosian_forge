import unittest
from traits.api import HasTraits, Int, List
def test_delete_step_slice(self):
    foo = MyClass()
    foo.l = [0, 1, 2, 3, 4]
    del foo.l[0:5:2]
    self.assertEqual(len(foo.l_events), 1)
    event, = foo.l_events
    self.assertEqual(event.index, slice(0, 5, 2))
    self.assertEqual(event.removed, [0, 2, 4])
    self.assertEqual(event.added, [])