import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
def test_ifelse_empty_one_branch(self):

    @njit
    def foo(k, v):
        d = Dict()
        if k:
            d[k] = v
        return d
    k, v = (123, 321)
    d = foo(k, v)
    self.assertEqual(dict(d), {k: v})
    k, v = (0, 0)
    d = foo(k, v)
    self.assertEqual(dict(d), {})
    self.assertEqual(typeof(d).key_type, typeof(k))
    self.assertEqual(typeof(d).value_type, typeof(v))