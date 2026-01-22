import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_object_copies(self):
    a = A()
    set1 = copy.deepcopy(a.aset)
    self.assertIsNone(set1.object())
    set1.add(10)
    self.assertEqual(len(a.events), 0)
    a.aset.add(20)
    self.assertEqual(len(a.events), 1)
    set2 = copy.deepcopy(set1)
    set2.add(30)
    self.assertIsNone(set2.object())
    set3 = a.aset.copy()
    self.assertIs(type(set3), set)
    set3.remove(20)