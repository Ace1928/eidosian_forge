import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_remove_error(self):
    self.disable_leak_check()
    pyfunc = remove_usecase
    cfunc = jit(nopython=True)(pyfunc)
    items = tuple(set(self.sparse_array(3)))
    a = items[1:]
    b = (items[0],)
    with self.assertRaises(KeyError):
        cfunc(a, b)