from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_utf8_truncate(self):
    """
        utf8_truncate()
        """
    from passlib.utils import utf8_truncate
    for source in [b'', b'1', b'123', b'\x1a', b'\x1a' * 10, b'\x7f', b'\x7f' * 10, b'a\xc2\xa0\xc3\xbe\xc3\xbe', b'abcdefghjusdfaoiu\xc2\xa0\xc3\xbe\xc3\xbedsfioauweoiruer']:
        source.decode('utf-8')
        end = len(source)
        for idx in range(end + 16):
            prefix = 'source=%r index=%r: ' % (source, idx)
            result = utf8_truncate(source, idx)
            result.decode('utf-8')
            self.assertLessEqual(len(result), end, msg=prefix)
            self.assertGreaterEqual(len(result), min(idx, end), msg=prefix)
            self.assertLess(len(result), min(idx + 4, end + 1), msg=prefix)
            self.assertEqual(result, source[:len(result)], msg=prefix)
    for source in [b'\xca', b'\xca' * 10, b'\x00', b'\x00' * 10]:
        end = len(source)
        for idx in range(end + 16):
            prefix = 'source=%r index=%r: ' % (source, idx)
            result = utf8_truncate(source, idx)
            self.assertEqual(result, source[:idx], msg=prefix)
    for source in [b'\xaa', b'\xaa' * 10]:
        end = len(source)
        for idx in range(end + 16):
            prefix = 'source=%r index=%r: ' % (source, idx)
            result = utf8_truncate(source, idx)
            self.assertEqual(result, source[:idx + 3], msg=prefix)
    source = b'MN\xff\xa0\xa1\xa2\xaaOP\xab'
    self.assertEqual(utf8_truncate(source, 0), b'')
    self.assertEqual(utf8_truncate(source, 1), b'M')
    self.assertEqual(utf8_truncate(source, 2), b'MN')
    self.assertEqual(utf8_truncate(source, 3), b'MN\xff\xa0\xa1\xa2')
    self.assertEqual(utf8_truncate(source, 4), b'MN\xff\xa0\xa1\xa2\xaa')
    self.assertEqual(utf8_truncate(source, 5), b'MN\xff\xa0\xa1\xa2\xaa')
    self.assertEqual(utf8_truncate(source, 6), b'MN\xff\xa0\xa1\xa2\xaa')
    self.assertEqual(utf8_truncate(source, 7), b'MN\xff\xa0\xa1\xa2\xaa')
    self.assertEqual(utf8_truncate(source, 8), b'MN\xff\xa0\xa1\xa2\xaaO')
    self.assertEqual(utf8_truncate(source, 9), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')
    self.assertEqual(utf8_truncate(source, 10), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')
    self.assertEqual(utf8_truncate(source, 11), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')