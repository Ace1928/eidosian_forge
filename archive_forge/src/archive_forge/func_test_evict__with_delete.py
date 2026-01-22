import unittest
from cachetools import MRUCache
from . import CacheTestMixin
def test_evict__with_delete(self):
    cache = MRUCache(maxsize=2)
    cache[1] = 1
    cache[2] = 2
    del cache[2]
    cache[3] = 3
    assert 2 not in cache
    assert 1 in cache
    cache[4] = 4
    assert 1 not in cache
    assert 3 in cache
    assert 4 in cache