import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_get_set_property(self):
    trait = CTrait(TraitKind.trait)
    self.assertIsNone(trait.property_fields)

    def value_get(self):
        return self.__dict__.get('_value', 0)

    def value_set(self, value):
        old_value = self.__dict__.get('_value', 0)
        if value != old_value:
            self._value = value
            self.trait_property_changed('value', old_value, value)
    trait.property_fields = (value_get, value_set, None)
    fget, fset, validate = trait.property_fields
    self.assertIs(fget, value_get)
    self.assertIs(fset, value_set)
    self.assertIsNone(validate)
    with self.assertRaises(TypeError):
        trait._get_property(fget)