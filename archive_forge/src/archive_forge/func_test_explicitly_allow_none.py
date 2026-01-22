import unittest
from traits.api import BaseInstance, HasStrictTraits, Instance, TraitError
def test_explicitly_allow_none(self):
    obj = HasSlices(none_explicitly_allowed=slice(2, 5))
    self.assertIsNotNone(obj.none_explicitly_allowed)
    obj.none_explicitly_allowed = None
    self.assertIsNone(obj.none_explicitly_allowed)
    obj = HasSlices(also_allow_none=slice(2, 5))
    self.assertIsNotNone(obj.also_allow_none)
    obj.also_allow_none = None
    self.assertIsNone(obj.also_allow_none)