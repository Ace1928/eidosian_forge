import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_sync_alias(self):
    """ Test synchronization of a trait with an aliased trait.
        """
    a = A()
    b = B()
    a.sync_trait('t', b, 'u')
    with self.assertTraitDoesNotChange(b, 't'):
        a.t = 5
    self.assertEqual(a.t, b.u)
    b.u = 7
    self.assertEqual(a.t, b.u)