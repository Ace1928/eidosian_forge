import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_dict_default_kind(self):
    a = A()
    dict_trait = a.traits()['adict']
    self.assertEqual(dict_trait.default_kind, 'dict')