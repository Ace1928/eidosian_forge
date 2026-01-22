import unittest
import cachetools.func
def test_decorator_clear(self):
    cached = self.decorator(maxsize=2)(lambda n: n)
    self.assertEqual(cached.cache_parameters(), {'maxsize': 2, 'typed': False})
    self.assertEqual(cached.cache_info(), (0, 0, 2, 0))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (0, 1, 2, 1))
    cached.cache_clear()
    self.assertEqual(cached.cache_info(), (0, 0, 2, 0))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (0, 1, 2, 1))