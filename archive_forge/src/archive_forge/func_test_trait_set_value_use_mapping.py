import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_set_value_use_mapping(self):
    obj = TraitWithMappingAndCallable(value=(0, 0, 0))
    self.assertEqual(obj.value, (0, 0, 0))
    self.assertEqual(obj.value_, 999)