import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@njit
def objmode_func():
    """
            Wrapper to call py_func from objmode. This tests
            large kws with objmode. If the definition for the
            call is not properly updated this test will fail.
            """
    with objmode(a='int64', b='int64', c='int64', d='int64', e='int64', f='int64', g='int64', h='int64', i='int64', j='int64', k='int64', l='int64', m='int64', n='int64', o='int64', p='int64'):
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = py_func()
    return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p