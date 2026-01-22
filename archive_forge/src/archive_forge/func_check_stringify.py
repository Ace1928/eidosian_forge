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
def check_stringify(self, strfn, prefix=False):
    nbd = Dict.empty(int32, int32)
    d = {}
    nbd[1] = 2
    d[1] = 2
    checker = self.assertIn if prefix else self.assertEqual
    checker(strfn(d), strfn(nbd))
    nbd[2] = 3
    d[2] = 3
    checker(strfn(d), strfn(nbd))
    for i in range(10, 20):
        nbd[i] = i + 1
        d[i] = i + 1
    checker(strfn(d), strfn(nbd))
    if prefix:
        self.assertTrue(strfn(nbd).startswith('DictType'))