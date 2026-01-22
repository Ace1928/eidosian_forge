from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_int12(self):
    engine = self.engine
    m = self.m
    self.check_int_pair(12, [(0, m(0, 0)), (63, m(0, 63) if engine.big else m(63, 0)), (4095, m(63, 63))])