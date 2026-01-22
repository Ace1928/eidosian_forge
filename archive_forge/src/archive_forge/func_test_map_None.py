from dulwich import lru_cache
from dulwich.tests import TestCase
def test_map_None(self):
    cache = lru_cache.LRUCache(max_cache=10)
    self.assertNotIn(None, cache)
    cache[None] = 1
    self.assertEqual(1, cache[None])
    cache[None] = 2
    self.assertEqual(2, cache[None])
    cache[1] = 3
    cache[None] = 1
    cache[None]
    cache[1]
    cache[None]
    self.assertEqual([None, 1], [n.key for n in cache._walk_lru()])