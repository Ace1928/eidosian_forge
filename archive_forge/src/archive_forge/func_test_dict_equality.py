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
def test_dict_equality(self):
    """
        Exercise dict.__eq__ and .__ne__
        """

    @njit
    def foo(na, nb, fa, fb):
        da = dictobject.new_dict(int32, float64)
        db = dictobject.new_dict(int32, float64)
        for i in range(na):
            da[i] = i * fa
        for i in range(nb):
            db[i] = i * fb
        return (da == db, da != db)
    self.assertEqual(foo(10, 10, 3, 3), (True, False))
    self.assertEqual(foo(10, 10, 3, 3.1), (False, True))
    self.assertEqual(foo(11, 10, 3, 3), (False, True))
    self.assertEqual(foo(10, 11, 3, 3), (False, True))