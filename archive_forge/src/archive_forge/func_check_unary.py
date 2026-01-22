import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def check_unary(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def check(arg):
        expected = pyfunc(arg)
        got = cfunc(arg)
        self.assertPreciseEqual(got, expected)
    return check