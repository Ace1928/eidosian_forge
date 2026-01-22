import random
import time
import unittest
def test_invalidate(self):
    cache = self._makeOne(3)
    cache.put('foo', 'bar')
    cache.put('FOO', 'BAR')
    cache.invalidate('foo')
    self.assertIsNone(cache.get('foo'))
    self.assertEqual(cache.get('FOO'), 'BAR')
    self.check_cache_is_consistent(cache)
    cache.invalidate('FOO')
    self.assertIsNone(cache.get('foo'))
    self.assertIsNone(cache.get('FOO'))
    self.assertEqual(cache.data, {})
    self.check_cache_is_consistent(cache)
    cache.put('foo', 'bar')
    cache.invalidate('nonexistingkey')
    self.assertEqual(cache.get('foo'), 'bar')
    self.assertIsNone(cache.get('FOO'))
    self.check_cache_is_consistent(cache)