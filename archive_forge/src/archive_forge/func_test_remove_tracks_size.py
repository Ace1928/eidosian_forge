from dulwich import lru_cache
from dulwich.tests import TestCase
def test_remove_tracks_size(self):
    cache = lru_cache.LRUSizeCache()
    self.assertEqual(0, cache._value_size)
    cache.add('my key', 'my value text')
    self.assertEqual(13, cache._value_size)
    node = cache._cache['my key']
    cache._remove_node(node)
    self.assertEqual(0, cache._value_size)