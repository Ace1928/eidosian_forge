import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_one_way_sync(self):
    """ Test one-way synchronization of two traits.
        """
    a = A(t=3)
    b = B(t=4)
    a.sync_trait('t', b, mutual=False)
    self.assertEqual(b.t, 3)
    a.t = 5
    self.assertEqual(b.t, a.t)
    with self.assertTraitDoesNotChange(a, 't'):
        b.t = 7
    a.sync_trait('t', b, remove=True)
    with self.assertTraitDoesNotChange(b, 't'):
        a.t = 12