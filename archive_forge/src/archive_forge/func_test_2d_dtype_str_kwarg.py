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
def test_2d_dtype_str_kwarg(self):
    pyfunc = self.pyfunc

    def func(m, n):
        return pyfunc((m, n), dtype='complex64')
    self.check_2d(func)