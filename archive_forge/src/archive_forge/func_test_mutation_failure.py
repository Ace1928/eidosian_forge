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
def test_mutation_failure(self):

    def setitem():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        ld['a'] = 12

    def delitem():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        del ld['a']

    def popitem():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        ld.popitem()

    def pop():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        ld.pop()

    def clear():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        ld.clear()

    def setdefault():
        ld = {'a': 1, 'b': 2j, 'c': 'd'}
        ld.setdefault('f', 1)
    illegals = (setitem, delitem, popitem, pop, clear, setdefault)
    for test in illegals:
        with self.subTest(test.__name__):
            with self.assertRaises(TypingError) as raises:
                njit(test)()
            expect = 'Cannot mutate a literal dictionary'
            self.assertIn(expect, str(raises.exception))