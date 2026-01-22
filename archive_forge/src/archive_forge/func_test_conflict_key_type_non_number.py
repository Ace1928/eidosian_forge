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
def test_conflict_key_type_non_number(self):

    @njit
    def foo(k1, v1, k2):
        d = Dict()
        d[k1] = v1
        return (d, d[k2])
    k1 = (np.int8(1), np.int8(2))
    k2 = (np.int32(1), np.int32(2))
    v1 = np.intp(123)
    with warnings.catch_warnings(record=True) as w:
        d, dk2 = foo(k1, v1, k2)
    self.assertEqual(len(w), 1)
    msg = 'unsafe cast from UniTuple(int32 x 2) to UniTuple(int8 x 2)'
    self.assertIn(msg, str(w[0]))
    keys = list(d.keys())
    self.assertEqual(keys[0], (1, 2))
    self.assertEqual(dk2, d[np.int32(1), np.int32(2)])