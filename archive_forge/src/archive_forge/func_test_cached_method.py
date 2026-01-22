import sys
import unittest
from Cython.Utils import (
def test_cached_method(self):
    obj = Cached()
    value = iter(range(3))
    cache = {(value,): 0}
    self.assertEqual(obj.cached_next(value), 0)
    self.set_of_names_equal(obj, {NAMES})
    self.assertEqual(getattr(obj, CACHE_NAME), cache)
    self.assertEqual(obj.cached_next(value), 0)
    self.set_of_names_equal(obj, {NAMES})
    self.assertEqual(getattr(obj, CACHE_NAME), cache)