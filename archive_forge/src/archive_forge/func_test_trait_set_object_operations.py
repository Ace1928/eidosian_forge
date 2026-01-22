import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_object_operations(self):
    a = A()
    a.aset.update({10: 'a'})
    self.assertEqual(a.aset, set([0, 1, 2, 3, 4, 10]))
    a.aset.intersection_update({3: 'b', 4: 'b', 10: 'a', 11: 'b'})
    self.assertEqual(a.aset, set([3, 4, 10]))
    a.aset.difference_update({10: 'a', 11: 'b'})
    self.assertEqual(a.aset, set([3, 4]))
    a.aset.symmetric_difference_update({10: 'a', 4: 'b'})
    self.assertEqual(a.aset, set([3, 10]))