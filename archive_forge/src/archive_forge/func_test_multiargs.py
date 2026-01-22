import random
import time
import unittest
def test_multiargs(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache)

    def moreargs(*args):
        return args
    decorated = decorator(moreargs)
    result = decorated(3, 4, 5)
    self.assertEqual(cache[3, 4, 5], (3, 4, 5))
    self.assertEqual(result, (3, 4, 5))
    self.assertEqual(len(cache), 1)