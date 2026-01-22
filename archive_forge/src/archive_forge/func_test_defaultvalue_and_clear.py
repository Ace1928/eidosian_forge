import random
import time
import unittest
def test_defaultvalue_and_clear(self):
    size = 10
    maker = self._makeOne(maxsize=size)
    for i in range(100):
        decorated = maker.lrucache()(_adder)
        decorated(10)
    self.assertEqual(len(maker._cache), 100)
    for _cache in maker._cache.values():
        self.assertEqual(_cache.size, size)
        self.assertEqual(len(_cache.data), 1)
    maker.clear()
    for _cache in maker._cache.values():
        self.assertEqual(_cache.size, size)
        self.assertEqual(len(_cache.data), 0)