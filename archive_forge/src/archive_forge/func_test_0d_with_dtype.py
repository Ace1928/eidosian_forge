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
def test_0d_with_dtype(self):

    def pyfunc(arg):
        return np.array(arg, dtype=np.int16)
    self.check_outputs(pyfunc, [(42,), (3.5,)])