import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_sync_lists_partial_slice(self):
    """ Test synchronization of list traits when there is a partial slice.

        Regression test for enthought/traits#540
        """
    a = A()
    b = B()
    a.sync_trait('l', b)
    a.l = [1, 2, 3]
    self.assertEqual(a.l, b.l)
    b.l = [4, 5]
    self.assertEqual(a.l, b.l)
    with self.assertTraitChanges(b, 'l_items'):
        a.l[:] = [7]
    self.assertEqual(b.l, [7])
    with self.assertTraitChanges(b, 'l_items'):
        a.l[:0] = [6]
    self.assertEqual(b.l, [6, 7])