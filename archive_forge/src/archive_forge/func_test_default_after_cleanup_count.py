from .. import fifo_cache, tests
def test_default_after_cleanup_count(self):
    c = fifo_cache.FIFOCache(5)
    self.assertEqual(4, c._after_cleanup_count)
    c[1] = 2
    c[2] = 3
    c[3] = 4
    c[4] = 5
    c[5] = 6
    self.assertEqual({1, 2, 3, 4, 5}, c.keys())
    c[6] = 7
    self.assertEqual({3, 4, 5, 6}, c.keys())