from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_encode_bytes_bad(self):
    """test encode_bytes() with bad input"""
    engine = self.engine
    encode = engine.encode_bytes
    self.assertRaises(TypeError, encode, u('\x00'))
    self.assertRaises(TypeError, encode, None)