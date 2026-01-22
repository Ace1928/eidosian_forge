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
def test_heterogeneous_but_castable_to_homogeneous(self):

    def bar(d):
        ...

    @overload(bar)
    def ol_bar(d):
        self.assertTrue(isinstance(d, types.DictType))
        self.assertEqual(d.initial_value, None)
        self.assertEqual(hasattr(d, 'literal_value'), False)
        return lambda d: d

    @njit
    def foo():
        x = {'a': 1j, 'b': 2, 'c': 3}
        bar(x)
    foo()