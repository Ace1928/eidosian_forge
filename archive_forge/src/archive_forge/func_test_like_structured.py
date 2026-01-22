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
@unittest.expectedFailure
def test_like_structured(self):
    dtype = np.dtype([('a', np.int16), ('b', np.float32)])

    def func(arr):
        return np.full_like(arr, 4.5)
    self.check_like(func, dtype)