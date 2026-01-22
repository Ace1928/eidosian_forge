from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def literal_iter_stopiteration_usecase():
    s = '大处着眼，小处着手。'
    i = iter(s)
    n = len(s)
    for _ in range(n + 1):
        next(i)