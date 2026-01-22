import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
def test_locked_nospace(self):
    cached = Locked(LRUCache(maxsize=0))
    self.assertEqual(cached.get(0), 1)
    self.assertEqual(cached.get(1), 3)
    self.assertEqual(cached.get(1), 5)
    self.assertEqual(cached.get(1.0), 7)
    self.assertEqual(cached.get(1.0), 9)