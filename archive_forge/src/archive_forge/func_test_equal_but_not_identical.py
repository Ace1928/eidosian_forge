import random
import time
import unittest
def test_equal_but_not_identical(self):
    cache = self._makeOne(1)
    tuple_one = (1, 1)
    tuple_two = (1, 1)
    cache.put(tuple_one, 42)
    self.assertEqual(cache.get(tuple_one), 42)
    self.assertEqual(cache.get(tuple_two), 42)
    self.check_cache_is_consistent(cache)
    cache = self._makeOne(1)
    cache.put(tuple_one, 42)
    cache.invalidate(tuple_two)
    self.assertIsNone(cache.get(tuple_one))
    self.assertIsNone(cache.get(tuple_two))