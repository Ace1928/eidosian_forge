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
def test_identity_dtype(self):
    for dtype in (np.complex64, np.int16, np.bool_, np.dtype('bool'), 'bool_'):

        def func(n):
            return np.identity(n, dtype)
        self.check_identity(func)