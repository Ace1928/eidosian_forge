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
def test_dict_pop_many(self):
    """
        Exercise dictionary .pop
        """

    @njit
    def core(d, pops):
        total = 0
        for k in pops:
            total += k + d.pop(k, 0.123) + len(d)
            total *= 2
        return total

    @njit
    def foo(keys, vals, pops):
        d = dictobject.new_dict(int32, float64)
        for k, v in zip(keys, vals):
            d[k] = v
        return core(d, pops)
    keys = [1, 2, 3]
    vals = [0.1, 0.2, 0.3]
    pops = [2, 3, 3, 1, 0, 2, 1, 0, -1]
    self.assertEqual(foo(keys, vals, pops), core.py_func(dict(zip(keys, vals)), pops))