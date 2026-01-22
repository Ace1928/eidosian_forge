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
def test_023_closure(self):

    @njit
    def foo():
        d = dictobject.new_dict(int32, float64)

        def bar():
            d[1] = 12.0
            d[2] = 14.0
        bar()
        return [x for x in d.keys()]
    self.assertEqual(foo(), [1, 2])