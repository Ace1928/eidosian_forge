from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_getrandbytes(self):
    """getrandbytes()"""
    from passlib.utils import getrandbytes
    wrapper = partial(getrandbytes, self.getRandom())
    self.assertEqual(len(wrapper(0)), 0)
    a = wrapper(10)
    b = wrapper(10)
    self.assertIsInstance(a, bytes)
    self.assertEqual(len(a), 10)
    self.assertEqual(len(b), 10)
    self.assertNotEqual(a, b)