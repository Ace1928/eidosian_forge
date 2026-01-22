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
def check_runtime_errors(self, cfunc, generate_starargs):
    self.assert_no_memory_leak()
    self.disable_leak_check()
    a, b, c, d, e = self._3d_arrays()
    with self.assert_invalid_sizes():
        args = next(generate_starargs())
        cfunc(a[:-1], b, c, *args)