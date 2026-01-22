import unittest
from traits.api import (
def test_set_event_repr(self):
    self.foo.aset.symmetric_difference_update({3, 4})
    event = self.foo.event
    event_str = 'TraitSetEvent(removed={3}, added={4})'
    self.assertEqual(repr(event), event_str)
    self.assertIsInstance(eval(repr(event)), TraitSetEvent)