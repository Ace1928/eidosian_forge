import random
import time
import unittest
def test_cache_attr(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache)

    def wrapped(key):
        return key
    decorated = decorator(wrapped)
    self.assertTrue(decorated._cache is cache)