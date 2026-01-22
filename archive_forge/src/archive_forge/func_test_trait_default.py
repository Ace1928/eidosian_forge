import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_default(self):
    obj = TraitWithMappingAndCallable()
    self.assertEqual(obj.value, 5)
    self.assertEqual(obj.value_, 5)