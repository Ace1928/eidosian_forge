import unittest
import cachetools.func
def test_decorator_user_function(self):
    cached = self.decorator(lambda n: n)
    self.assertEqual(cached.cache_parameters(), {'maxsize': 128, 'typed': False})
    self.assertEqual(cached.cache_info(), (0, 0, 128, 0))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (0, 1, 128, 1))
    self.assertEqual(cached(1), 1)
    self.assertEqual(cached.cache_info(), (1, 1, 128, 1))
    self.assertEqual(cached(1.0), 1.0)
    self.assertEqual(cached.cache_info(), (2, 1, 128, 1))