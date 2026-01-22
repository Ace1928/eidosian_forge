import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
def test_locked_nocache(self):
    cached = Locked(None)
    self.assertEqual(cached.get(0), 0)
    self.assertEqual(cached.get(1), 0)
    self.assertEqual(cached.get(1), 0)
    self.assertEqual(cached.get(1.0), 0)
    self.assertEqual(cached.get(1.0), 0)