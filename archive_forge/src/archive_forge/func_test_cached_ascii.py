from .. import cache_utf8
from . import TestCase
def test_cached_ascii(self):
    x = b'%s %s' % (b'simple', b'text')
    y = b'%s %s' % (b'simple', b'text')
    self.assertIsNot(x, y)
    xp = cache_utf8.get_cached_ascii(x)
    yp = cache_utf8.get_cached_ascii(y)
    self.assertIs(xp, x)
    self.assertIs(xp, yp)
    uni_x = cache_utf8.decode(x)
    self.assertEqual('simple text', uni_x)
    self.assertIsInstance(uni_x, str)
    utf8_x = cache_utf8.encode(uni_x)
    self.assertIs(utf8_x, x)