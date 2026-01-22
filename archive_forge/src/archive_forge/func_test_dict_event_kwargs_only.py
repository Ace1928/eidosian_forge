import unittest
from traits.api import (
def test_dict_event_kwargs_only(self):
    with self.assertRaises(TypeError):
        TraitDictEvent({}, {'black': 0}, {'blue': 2})