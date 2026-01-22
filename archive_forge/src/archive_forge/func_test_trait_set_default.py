import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_default(self):
    a = A()
    set_trait = a.traits()['aset']
    self.assertEqual(set_trait.default, {0, 1, 2, 3, 4})
    set_trait.default.remove(2)
    self.assertEqual(a.aset, {0, 1, 2, 3, 4})