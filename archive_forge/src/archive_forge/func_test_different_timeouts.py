import random
import time
import unittest
def test_different_timeouts(self):
    cache = self._makeOne(3, default_timeout=0.1)
    cache.put('one', 1)
    cache.put('two', 2, timeout=0.2)
    cache.put('three', 3, timeout=0.3)
    self.assertEqual(cache.get('one'), 1)
    self.assertEqual(cache.get('two'), 2)
    self.assertEqual(cache.get('three'), 3)
    time.sleep(0.1)
    self.assertIsNone(cache.get('one'))
    self.assertEqual(cache.get('two'), 2)
    self.assertEqual(cache.get('three'), 3)
    time.sleep(0.1)
    self.assertIsNone(cache.get('one'))
    self.assertIsNone(cache.get('two'))
    self.assertEqual(cache.get('three'), 3)
    time.sleep(0.1)
    self.assertIsNone(cache.get('one'))
    self.assertIsNone(cache.get('two'))
    self.assertIsNone(cache.get('three'))
    self.check_cache_is_consistent(cache)