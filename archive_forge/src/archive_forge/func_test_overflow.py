from dulwich import lru_cache
from dulwich.tests import TestCase
def test_overflow(self):
    """Adding extra entries will pop out old ones."""
    cache = lru_cache.LRUCache(max_cache=1, after_cleanup_count=1)
    cache['foo'] = 'bar'
    cache['baz'] = 'biz'
    self.assertNotIn('foo', cache)
    self.assertIn('baz', cache)
    self.assertEqual('biz', cache['baz'])