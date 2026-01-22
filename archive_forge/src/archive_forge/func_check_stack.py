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
def check_stack(self, pyfunc, cfunc, args):
    expected = pyfunc(*args)
    got = cfunc(*args)
    self.assertEqual(got.shape, expected.shape)
    self.assertPreciseEqual(got.flatten(), expected.flatten())