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
def test_018_keys_iter_are_views(self):

    @njit
    def foo():
        d = dictobject.new_dict(int32, float64)
        d[11] = 12.0
        k1 = d.keys()
        d[22] = 9.0
        k2 = d.keys()
        rk1 = [x for x in k1]
        rk2 = [x for x in k2]
        return (rk1, rk2)
    a, b = foo()
    self.assertEqual(a, b)
    self.assertEqual(a, [11, 22])