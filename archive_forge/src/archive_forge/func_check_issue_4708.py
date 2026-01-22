from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@from_generic([transpose_issue_4708])
def check_issue_4708(pyfunc, m, n):
    expected = pyfunc.py_func(m, n)
    got = pyfunc(m, n)
    np.testing.assert_equal(got, expected)