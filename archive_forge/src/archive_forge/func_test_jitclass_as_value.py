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
def test_jitclass_as_value(self):

    @njit
    def foo(x):
        d = Dict()
        d[0] = x
        d[1] = Bag(101)
        return d
    d = foo(Bag(a=100))
    self.assertEqual(d[0].a, 100)
    self.assertEqual(d[1].a, 101)