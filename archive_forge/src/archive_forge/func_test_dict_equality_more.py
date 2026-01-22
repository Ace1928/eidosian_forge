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
def test_dict_equality_more(self):
    """
        Exercise dict.__eq__
        """

    @njit
    def foo(ak, av, bk, bv):
        da = dictobject.new_dict(int32, float64)
        db = dictobject.new_dict(int64, float32)
        for i in range(len(ak)):
            da[ak[i]] = av[i]
        for i in range(len(bk)):
            db[bk[i]] = bv[i]
        return da == db
    ak = [1, 2, 3]
    av = [2, 3, 4]
    bk = [1, 2, 3]
    bv = [2, 3, 4]
    self.assertTrue(foo(ak, av, bk, bv))
    ak = [1, 2, 3]
    av = [2, 3, 4]
    bk = [1, 2, 2, 3]
    bv = [2, 1, 3, 4]
    self.assertTrue(foo(ak, av, bk, bv))
    ak = [1, 2, 3]
    av = [2, 3, 4]
    bk = [1, 2, 3]
    bv = [2, 1, 4]
    self.assertFalse(foo(ak, av, bk, bv))
    ak = [0, 2, 3]
    av = [2, 3, 4]
    bk = [1, 2, 3]
    bv = [2, 3, 4]
    self.assertFalse(foo(ak, av, bk, bv))