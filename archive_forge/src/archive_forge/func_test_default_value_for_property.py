import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_default_value_for_property(self):
    """ Don't segfault when calling default_value_for on a Property trait.
        """
    y_trait = SimpleProperty.class_traits()['y']
    simple_property = SimpleProperty()
    self.assertIsNone(y_trait.default_value_for(simple_property, 'y'))