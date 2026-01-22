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
def test_dict_get_with_default(self):
    """
        Exercise dict.get(k, d) where d is set
        """

    @njit
    def foo(n, target, default):
        d = dictobject.new_dict(int32, float64)
        for i in range(n):
            d[i] = i
        return d.get(target, default)
    self.assertEqual(foo(5, 3, -1), 3)
    self.assertEqual(foo(5, 5, -1), -1)