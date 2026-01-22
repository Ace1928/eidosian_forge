from dulwich import lru_cache
from dulwich.tests import TestCase
def test_adding_clears_cache_based_on_size(self):
    """The cache is cleared in LRU order until small enough."""
    cache = lru_cache.LRUSizeCache(max_size=20)
    cache.add('key1', 'value')
    cache.add('key2', 'value2')
    cache.add('key3', 'value23')
    self.assertEqual(5 + 6 + 7, cache._value_size)
    cache['key2']
    cache.add('key4', 'value234')
    self.assertEqual(6 + 8, cache._value_size)
    self.assertEqual({'key2': 'value2', 'key4': 'value234'}, cache.items())