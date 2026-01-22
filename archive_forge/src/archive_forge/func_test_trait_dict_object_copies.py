import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_dict_object_copies(self):
    a = A()
    dict = copy.deepcopy(a.adict)
    self.assertIsNone(dict.object())
    dict['key'] = 10
    self.assertEqual(len(a.events), 0)
    a.adict['key'] = 10
    self.assertEqual(len(a.events), 1)
    dict2 = copy.deepcopy(dict)
    dict2['key2'] = 20
    self.assertIsNone(dict2.object())