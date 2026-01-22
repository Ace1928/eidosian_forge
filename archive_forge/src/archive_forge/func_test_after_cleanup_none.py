from dulwich import lru_cache
from dulwich.tests import TestCase
def test_after_cleanup_none(self):
    cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=None)
    self.assertEqual(4, cache._after_cleanup_count)