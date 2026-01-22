import random
import time
import unittest
def test_named_cache(self):
    maker = self._makeOne()
    size = 10
    name = 'name'
    decorated = maker.lrucache(maxsize=size, name=name)(_adder)
    self.assertEqual(list(maker._cache.keys()), [name])
    self.assertEqual(maker._cache[name].size, size)
    decorated(10)
    decorated(11)
    self.assertEqual(len(maker._cache[name].data), 2)