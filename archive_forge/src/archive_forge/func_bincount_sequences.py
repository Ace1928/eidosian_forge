import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def bincount_sequences(self):
    """
        Some test sequences for np.bincount()
        """
    a = [1, 2, 5, 2, 3, 20]
    b = np.array([5, 8, 42, 5])
    c = self.rnd.randint(0, 100, size=300).astype(np.int8)
    return (a, b, c)