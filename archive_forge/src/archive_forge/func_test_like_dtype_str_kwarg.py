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
def test_like_dtype_str_kwarg(self):

    def func(arr):
        return np.full_like(arr, 4.5, 'bool_')
    self.check_like(func, np.float64)