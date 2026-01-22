import random
import time
import unittest
def test_ctor_w_size_w_timeout(self):
    from repoze.lru import ExpiringLRUCache
    decorator = self._makeOne(maxsize=10, timeout=30)
    self.assertIsInstance(decorator.cache, ExpiringLRUCache)
    self.assertEqual(decorator.cache.size, 10)
    self.assertEqual(decorator.cache.default_timeout, 30)