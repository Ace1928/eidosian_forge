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
def test_unify_across_function_call(self):

    @njit
    def bar(x):
        o = {1: 2}
        if x:
            o = {2: 3}
        return o

    @njit
    def foo(x):
        if x:
            d = {3: 4}
        else:
            d = bar(x)
        return d
    e1 = Dict()
    e1[3] = 4
    e2 = Dict()
    e2[1] = 2
    self.assertEqual(foo(True), e1)
    self.assertEqual(foo(False), e2)