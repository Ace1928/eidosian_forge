import unittest
from cachetools import MRUCache
from . import CacheTestMixin
def test_evict__writes_only(self):
    cache = MRUCache(maxsize=2)
    cache[1] = 1
    cache[2] = 2
    cache[3] = 3
    assert len(cache) == 2
    assert 1 not in cache, "Wrong key was evicted. Should have been '1'."
    assert 2 in cache
    assert 3 in cache