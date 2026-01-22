import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Any
def test_with_default_value_and_factory(self):
    with self.assertRaises(TypeError):
        Any(23, factory=int)