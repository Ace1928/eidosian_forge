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
def test_repeated_key_literal_value(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        self.assertEqual(x.literal_value, {types.literal('a'): types.literal('aaaa'), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d')})

        def impl(x):
            pass
        return impl

    @njit
    def foo():
        ld = {'a': 1, 'a': 10, 'b': 2j, 'c': 'd', 'a': 'aaaa'}
        bar(ld)
    foo()