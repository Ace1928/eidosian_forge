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
def test_dict_as_arg(self):

    @njit
    def bar(fake_kwargs=None):
        if fake_kwargs is not None:
            fake_kwargs['d'][:] += 10

    @njit
    def foo():
        a = 1
        b = 2j
        c = 'string'
        d = np.zeros(3)
        e = {'a': a, 'b': b, 'c': c, 'd': d}
        bar(fake_kwargs=e)
        return e['d']
    np.testing.assert_allclose(foo(), np.ones(3) * 10)