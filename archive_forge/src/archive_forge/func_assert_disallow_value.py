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
def assert_disallow_value(self, ty):
    msg = '{} as value is forbidden'.format(ty)
    self.assert_disallow(msg, lambda: Dict.empty(types.intp, ty))

    @njit
    def foo():
        Dict.empty(types.intp, ty)
    self.assert_disallow(msg, foo)