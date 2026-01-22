import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def generate_advanced_indices(self, N, many=True):
    choices = [np.int16([0, N - 1, -2])]
    if many:
        choices += [np.uint16([0, 1, N - 1]), np.bool_([0, 1, 1, 0])]
    return choices