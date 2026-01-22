import unittest
from traits.api import (
def test_set_event_kwargs_only(self):
    with self.assertRaises(TypeError):
        TraitSetEvent({3}, {4})