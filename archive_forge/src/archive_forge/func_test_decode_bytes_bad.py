from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_decode_bytes_bad(self):
    """test decode_bytes() with bad input"""
    engine = self.engine
    decode = engine.decode_bytes
    self.assertRaises(ValueError, decode, engine.bytemap[:5])
    self.assertTrue(self.bad_byte not in engine.bytemap)
    self.assertRaises(ValueError, decode, self.bad_byte * 4)
    self.assertRaises(TypeError, decode, engine.charmap[:4])
    self.assertRaises(TypeError, decode, None)