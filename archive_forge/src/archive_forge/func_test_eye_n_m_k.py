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
def test_eye_n_m_k(self):

    def func(n, m, k):
        return np.eye(n, m, k)
    self.check_eye_n_m_k(func)