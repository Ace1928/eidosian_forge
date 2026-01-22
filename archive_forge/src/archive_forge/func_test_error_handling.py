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
def test_error_handling(self):
    d = np.array([[[1.0]]])
    cfunc = nrtjit(self.py)
    with self.assertRaises(TypeError):
        cfunc()
    with self.assertRaises(TypingError):
        cfunc(d)
    with self.assertRaises(TypingError):
        dfunc = nrtjit(self.py_kw)
        dfunc(d, k=3)