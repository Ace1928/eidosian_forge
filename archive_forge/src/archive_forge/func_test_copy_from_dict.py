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
def test_copy_from_dict(self):
    expect = {k: float(v) for k, v in zip(range(10), range(10, 20))}
    nbd = Dict.empty(int32, float64)
    for k, v in expect.items():
        nbd[k] = v
    got = dict(nbd)
    self.assertEqual(got, expect)