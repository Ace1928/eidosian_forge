import pickle
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
def test_in_add(self):
    a = OrderedSet()
    self.assertNotIn(1, a)
    self.assertNotIn(None, a)
    a.add(None)
    self.assertNotIn(1, a)
    self.assertIn(None, a)
    a.add(1)
    self.assertIn(1, a)
    self.assertIn(None, a)
    a.add(0)
    self.assertEqual(list(a), [None, 1, 0])
    a.add(1)
    self.assertEqual(list(a), [None, 1, 0])