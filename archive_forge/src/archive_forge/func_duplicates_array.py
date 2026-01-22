import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def duplicates_array(self, n):
    """
        Get a 1d array with many duplicate values.
        """
    a = self._range(np.sqrt(n))
    return self._random_choice(a, n)