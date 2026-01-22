import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_default_keyword_only(self):
    with self.assertRaises(TypeError):
        PrefixMap({'yes': 1, 'no': 0}, 'yes')