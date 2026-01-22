import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_values_is_empty(self):
    with self.assertRaises(ValueError):
        PrefixList([])