import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
def test_nospace(self):
    cached = Cached(LRUCache(maxsize=0))
    self.assertEqual(cached.get(0), 0)
    self.assertEqual(cached.get(1), 1)
    self.assertEqual(cached.get(1), 2)
    self.assertEqual(cached.get(1.0), 3)
    self.assertEqual(cached.get(1.0), 4)