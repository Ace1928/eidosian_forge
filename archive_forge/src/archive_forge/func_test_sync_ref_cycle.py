import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_sync_ref_cycle(self):
    """ Regression test for #69.
        """
    a = A()
    b = B()
    change_counter = [0]

    def _handle_change():
        change_counter[0] += 1
    b.on_trait_change(_handle_change, 't')
    a.sync_trait('t', b)
    a.t = 17
    self.assertEqual(change_counter, [1])
    del b
    a.t = 42
    self.assertEqual(change_counter, [1])