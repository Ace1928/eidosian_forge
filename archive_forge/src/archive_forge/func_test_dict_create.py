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
def test_dict_create(self):
    """
        Exercise dictionary creation, insertion and len
        """

    @njit
    def foo(n):
        d = dictobject.new_dict(int32, float32)
        for i in range(n):
            d[i] = i + 1
        return len(d)
    self.assertEqual(foo(n=0), 0)
    self.assertEqual(foo(n=1), 1)
    self.assertEqual(foo(n=2), 2)
    self.assertEqual(foo(n=100), 100)