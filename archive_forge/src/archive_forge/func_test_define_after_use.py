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
def test_define_after_use(self):

    @njit
    def foo(define):
        d = Dict()
        ct = len(d)
        for k, v in d.items():
            ct += v
        if define:
            d[1] = 2
        return (ct, d, len(d))
    ct, d, n = foo(True)
    self.assertEqual(ct, 0)
    self.assertEqual(n, 1)
    self.assertEqual(dict(d), {1: 2})
    ct, d, n = foo(False)
    self.assertEqual(ct, 0)
    self.assertEqual(dict(d), {})
    self.assertEqual(n, 0)