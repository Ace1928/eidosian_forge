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
def test_eye_n_m(self):

    def func(n, m):
        return np.eye(n, m)
    self.check_outputs(func, [(1, 2), (3, 2), (0, 3)])