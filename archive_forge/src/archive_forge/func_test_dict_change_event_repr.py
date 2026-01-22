import unittest
from traits.observation._dict_change_event import (
from traits.trait_dict_object import TraitDict
def test_dict_change_event_repr(self):
    event = DictChangeEvent(object=dict(), added={1: 1}, removed={'2': 2})
    actual = repr(event)
    self.assertEqual(actual, "DictChangeEvent(object={}, removed={'2': 2}, added={1: 1})")