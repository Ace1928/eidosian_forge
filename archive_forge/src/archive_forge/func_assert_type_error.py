import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
@contextlib.contextmanager
def assert_type_error(self, msg):
    with self.assertRaises(TypeError) as raises:
        yield
    if msg is not None:
        self.assertRegex(str(raises.exception), msg)