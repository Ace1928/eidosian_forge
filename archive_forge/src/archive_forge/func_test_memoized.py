import random
import time
import unittest
def test_memoized(self):
    from repoze.lru import lru_cache
    from repoze.lru import UnboundedCache
    maker = self._makeOne(maxsize=10)
    memo = maker.memoized('test')
    self.assertIsInstance(memo, lru_cache)
    self.assertIsInstance(memo.cache, UnboundedCache)
    self.assertIs(memo.cache, maker._cache['test'])