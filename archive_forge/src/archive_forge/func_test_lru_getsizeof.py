import unittest
from cachetools import LRUCache
from . import CacheTestMixin
def test_lru_getsizeof(self):
    cache = LRUCache(maxsize=3, getsizeof=lambda x: x)
    cache[1] = 1
    cache[2] = 2
    self.assertEqual(len(cache), 2)
    self.assertEqual(cache[1], 1)
    self.assertEqual(cache[2], 2)
    cache[3] = 3
    self.assertEqual(len(cache), 1)
    self.assertEqual(cache[3], 3)
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    with self.assertRaises(ValueError):
        cache[4] = 4
    self.assertEqual(len(cache), 1)
    self.assertEqual(cache[3], 3)