import unittest
import cachetools.func
def test_decorator_unbound(self):
    cached = self.decorator(maxsize=None)(lambda n: n)
    self.assertEqual(cached.cache_parameters(), {'maxsize': None, 'typed': False})
    self.assertEqual(cached.cache_info(), (0, 0, None, 0))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (0, 1, None, 1))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (1, 1, None, 1))
    self.assertEqual(cached(1.0), 1.0)
    self.assertEqual(cached.cache_info(), (2, 1, None, 1))