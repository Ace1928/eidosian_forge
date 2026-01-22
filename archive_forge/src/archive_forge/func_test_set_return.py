import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_set_return(self):
    pyfunc = set_return_usecase
    cfunc = jit(nopython=True)(pyfunc)
    arg = self.duplicates_array(200)
    self.assertEqual(cfunc(arg), set(arg))