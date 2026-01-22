import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_empty_map(self):
    with self.assertRaises(ValueError):
        PrefixMap({})