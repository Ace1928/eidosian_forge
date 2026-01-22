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
def inputs13():
    yield (np.array([1, 1 + 256]), 255)
    yield (np.array([0, 75, 150, 225, 300]), 255)
    yield (np.array([0, 1, 2, -1, 0]), 4)
    yield (np.array([2, 3, 4, 5, 2, 3, 4, 5]), 4)
    yield (wrap_uneven, 250)