from .. import fifo_cache, tests
def test_adding_large_key(self):
    c = fifo_cache.FIFOSizeCache(10, 8)
    c[1] = 'abcdefgh'
    self.assertEqual({}, c)
    c[1] = 'abcdefg'
    self.assertEqual({1: 'abcdefg'}, c)
    c[1] = 'abcdefgh'
    self.assertEqual({}, c)
    self.assertEqual(0, c._value_size)