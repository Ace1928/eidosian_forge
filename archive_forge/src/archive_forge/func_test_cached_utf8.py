from .. import cache_utf8
from . import TestCase
def test_cached_utf8(self):
    x = 'µyyåzz'.encode()
    y = 'µyyåzz'.encode()
    self.assertFalse(x is y)
    xp = cache_utf8.get_cached_utf8(x)
    yp = cache_utf8.get_cached_utf8(y)
    self.assertIs(xp, x)
    self.assertIs(xp, yp)