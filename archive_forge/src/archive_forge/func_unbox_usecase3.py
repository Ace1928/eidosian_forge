import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def unbox_usecase3(x):
    """
    Expect a (number, set of numbers) tuple.
    """
    a, b = x
    res = a
    for v in b:
        res += v
    return res