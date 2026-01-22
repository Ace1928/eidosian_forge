import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_build_set_nopython(self):
    arg = list(self.sparse_array(50))
    pyfunc = set_literal_convert_usecase(arg)
    cfunc = jit(nopython=True)(pyfunc)
    expected = pyfunc()
    got = cfunc()
    self.assertPreciseEqual(sorted(expected), sorted(got))