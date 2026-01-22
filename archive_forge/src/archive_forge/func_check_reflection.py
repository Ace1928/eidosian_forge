import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def check_reflection(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    samples = [(set([1.0, 2.0, 3.0, 4.0]), set([0.0])), (set([1.0, 2.0, 3.0, 4.0]), set([5.0, 6.0, 7.0, 8.0, 9.0]))]
    for dest, src in samples:
        expected = set(dest)
        got = set(dest)
        pyres = pyfunc(expected, src)
        with self.assertRefCount(got, src):
            cres = cfunc(got, src)
            self.assertPreciseEqual(cres, pyres)
            self.assertPreciseEqual(expected, got)
            self.assertEqual(pyres[0] is expected, cres[0] is got)
            del pyres, cres