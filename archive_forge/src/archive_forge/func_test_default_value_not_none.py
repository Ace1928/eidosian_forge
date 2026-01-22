import unittest
from traits.api import HasTraits, TraitError
from traits.trait_types import _NoneTrait
def test_default_value_not_none(self):
    with self.assertRaises(ValueError):

        class TestClass(HasTraits):
            none_trait = _NoneTrait(default_value=[])