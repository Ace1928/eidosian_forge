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
def test_issue6570_alignment_padding(self):
    keyty = types.Tuple([types.uint64, types.float32])

    @njit
    def foo():
        d = dictobject.new_dict(keyty, float64)
        t1 = np.array([3], dtype=np.uint64)
        t2 = np.array([5.67], dtype=np.float32)
        v1 = np.array([10.23], dtype=np.float32)
        d[t1[0], t2[0]] = v1[0]
        return (t1[0], t2[0]) in d
    self.assertTrue(foo())