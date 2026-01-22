from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_slice_passing(self):
    """
        Check passing a slice object to a Numba function.
        """

    def check(a, b, c, d, e, f):
        sl = slice(a, b, c)
        got = cfunc(sl)
        self.assertPreciseEqual(got, (d, e, f))
    maxposint = sys.maxsize
    maxnegint = -maxposint - 1
    cfunc = jit(nopython=True)(slice_passing)
    start_cases = [(None, 0), (42, 42), (-1, -1)]
    stop_cases = [(None, maxposint), (9, 9), (-11, -11)]
    step_cases = [(None, 1), (12, 12)]
    for (a, d), (b, e), (c, f) in itertools.product(start_cases, stop_cases, step_cases):
        check(a, b, c, d, e, f)
    start_cases = [(None, maxposint), (42, 42), (-1, -1)]
    stop_cases = [(None, maxnegint), (9, 9), (-11, -11)]
    step_cases = [(-1, -1), (-12, -12)]
    for (a, d), (b, e), (c, f) in itertools.product(start_cases, stop_cases, step_cases):
        check(a, b, c, d, e, f)
    with self.assertRaises(TypeError):
        cfunc(slice(1.5, 1, 1))