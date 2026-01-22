from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_encode_transposed_bytes(self):
    """test encode_transposed_bytes()"""
    engine = self.engine
    for result, input, offsets in self.transposed + self.transposed_dups:
        tmp = engine.encode_transposed_bytes(input, offsets)
        out = engine.decode_bytes(tmp)
        self.assertEqual(out, result)
    self.assertRaises(TypeError, engine.encode_transposed_bytes, u('a'), [])