import unittest
import cachetools
import cachetools.keys
def test_zero_size_cache_decorator(self):
    cache = self.cache(0)
    wrapper = cachetools.cached(cache)(self.func)
    self.assertEqual(len(cache), 0)
    self.assertEqual(wrapper.__wrapped__, self.func)
    self.assertEqual(wrapper(0), 0)
    self.assertEqual(len(cache), 0)