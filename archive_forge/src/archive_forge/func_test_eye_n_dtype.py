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
def test_eye_n_dtype(self):
    for dt in (None, np.complex128, np.complex64(1)):

        def func(n, dtype=dt):
            return np.eye(n, dtype=dtype)
        self.check_outputs(func, [(1,), (3,)])