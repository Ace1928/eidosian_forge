import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_trait_set_object_inplace(self):
    a = A()
    a.aset |= set([10])
    self.assertEqual(a.aset, set([0, 1, 2, 3, 4, 10]))
    a.aset &= set([3, 4, 10, 11])
    self.assertEqual(a.aset, set([3, 4, 10]))
    a.aset -= set([10, 11])
    self.assertEqual(a.aset, set([3, 4]))
    a.aset ^= set([10, 4])
    self.assertEqual(a.aset, set([3, 10]))