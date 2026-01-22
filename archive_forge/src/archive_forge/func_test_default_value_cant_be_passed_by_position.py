import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_default_value_cant_be_passed_by_position(self):
    with self.assertRaises(TypeError):
        PrefixList(['zero', 'one', 'two'], 'one')