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
def test_conflicting_key_type(self):

    @njit
    def foo(k, h, v):
        d = Dict()
        d[k] = v
        d[h] = v
        return d
    k, h, v = (123, 123.1, 321)
    with self.assertRaises(TypingError) as raises:
        foo(k, h, v)
    self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))