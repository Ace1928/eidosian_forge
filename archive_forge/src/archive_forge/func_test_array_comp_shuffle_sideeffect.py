import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def test_array_comp_shuffle_sideeffect(self):
    nelem = 100

    @jit(nopython=True)
    def foo():
        numbers = np.array([i for i in range(nelem)])
        np.random.shuffle(numbers)
        print(numbers)
    with captured_stdout() as gotbuf:
        foo()
    got = gotbuf.getvalue().strip()
    with captured_stdout() as expectbuf:
        print(np.array([i for i in range(nelem)]))
    expect = expectbuf.getvalue().strip()
    self.assertNotEqual(got, expect)
    self.assertRegex(got, '\\[(\\s*\\d+)+\\]')