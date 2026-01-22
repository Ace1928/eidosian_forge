from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_to_native_str(self):
    """test to_native_str()"""
    from passlib.utils import to_native_str
    self.assertEqual(to_native_str(u('abc'), 'ascii'), 'abc')
    self.assertEqual(to_native_str(b'abc', 'ascii'), 'abc')
    if PY3:
        self.assertEqual(to_native_str(u('à'), 'ascii'), 'à')
        self.assertRaises(UnicodeDecodeError, to_native_str, b'\xc3\xa0', 'ascii')
    else:
        self.assertRaises(UnicodeEncodeError, to_native_str, u('à'), 'ascii')
        self.assertEqual(to_native_str(b'\xc3\xa0', 'ascii'), 'Ã\xa0')
    self.assertEqual(to_native_str(u('à'), 'latin-1'), 'à')
    self.assertEqual(to_native_str(b'\xe0', 'latin-1'), 'à')
    self.assertEqual(to_native_str(u('à'), 'utf-8'), 'à' if PY3 else 'Ã\xa0')
    self.assertEqual(to_native_str(b'\xc3\xa0', 'utf-8'), 'à' if PY3 else 'Ã\xa0')
    self.assertRaises(TypeError, to_native_str, None, 'ascii')