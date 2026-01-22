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
def test_1d_with_non_const_str_dtype(self):

    @njit
    def func(arg, dt):
        return np.array(arg, dtype=dt)
    with self.assertRaises(TypingError) as raises:
        func((5, 3), 'int32')
    excstr = str(raises.exception)
    msg = f'If np.array dtype is a string it must be a string constant.'
    self.assertIn(msg, excstr)