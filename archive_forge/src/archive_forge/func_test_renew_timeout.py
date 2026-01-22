import random
import time
import unittest
def test_renew_timeout(self):
    cache = self._makeOne(3, default_timeout=0.2)
    cache.put('foo', 'bar')
    cache.put('foo2', 'bar2', timeout=10)
    cache.put('foo3', 'bar3', timeout=10)
    time.sleep(0.1)
    self.assertEqual(cache.get('foo'), 'bar')
    self.assertEqual(cache.get('foo2'), 'bar2')
    self.assertEqual(cache.get('foo3'), 'bar3')
    self.check_cache_is_consistent(cache)
    cache.put('foo', 'bar')
    cache.put('foo2', 'bar2', timeout=0.1)
    cache.put('foo3', 'bar3')
    time.sleep(0.1)
    self.assertEqual(cache.get('foo'), 'bar')
    self.assertIsNone(cache.get('foo2'))
    self.assertEqual(cache.get('foo3'), 'bar3')
    self.check_cache_is_consistent(cache)