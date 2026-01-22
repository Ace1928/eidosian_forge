import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_2d_shape_dtypes(self):

    def func1(m, n):
        return np.full((np.int16(m), np.int32(n)), 4.5)
    self.check_2d(func1)

    def func2(m, n):
        return np.full((np.int64(m), np.int8(n)), 4.5)
    self.check_2d(func2)
    if config.IS_32BITS:
        cfunc = nrtjit(lambda m, n: np.full((m, n), 4.5))
        with self.assertRaises(ValueError):
            cfunc(np.int64(1 << 32 - 1), 1)