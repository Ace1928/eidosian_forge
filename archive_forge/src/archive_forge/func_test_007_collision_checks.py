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
@unittest.skipUnless(sys.maxsize > 2 ** 32, '64 bit test only')
def test_007_collision_checks(self):

    @njit
    def foo(v1, v2):
        d = dictobject.new_dict(int64, float64)
        c1 = np.uint64(2 ** 61 - 1)
        c2 = np.uint64(0)
        assert hash(c1) == hash(c2)
        d[c1] = v1
        d[c2] = v2
        return (d[c1], d[c2])
    a, b = (10.0, 20.0)
    x, y = foo(a, b)
    self.assertEqual(x, a)
    self.assertEqual(y, b)