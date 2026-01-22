import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_set_of_tuples_inside_tuple(self):
    check = self.check_unary(unbox_usecase4)
    check((1, set([(2,), (3,)])))