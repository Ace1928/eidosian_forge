from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_memoized_property(self):
    from passlib.utils.decor import memoized_property

    class dummy(object):
        counter = 0

        @memoized_property
        def value(self):
            value = self.counter
            self.counter = value + 1
            return value
    d = dummy()
    self.assertEqual(d.value, 0)
    self.assertEqual(d.value, 0)
    self.assertEqual(d.counter, 1)
    prop = dummy.value
    if not PY3:
        self.assertIs(prop.im_func, prop.__func__)