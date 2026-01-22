from .. import fifo_cache, tests
def test_cleanup_at_deconstruct(self):
    log = []

    def logging_cleanup(key, value):
        log.append((key, value))
    c = fifo_cache.FIFOCache()
    c.add(1, 2, cleanup=logging_cleanup)
    del c
    self.assertEqual([], log)