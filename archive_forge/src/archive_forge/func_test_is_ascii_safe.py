from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_is_ascii_safe(self):
    """test is_ascii_safe()"""
    from passlib.utils import is_ascii_safe
    self.assertTrue(is_ascii_safe(b'\x00abc\x7f'))
    self.assertTrue(is_ascii_safe(u('\x00abc\x7f')))
    self.assertFalse(is_ascii_safe(b'\x00abc\x80'))
    self.assertFalse(is_ascii_safe(u('\x00abc\x80')))