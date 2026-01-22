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
def test_unwrap_basic(self):
    pyfunc = unwrap
    cfunc = njit(pyfunc)
    pyfunc1 = unwrap1
    cfunc1 = njit(pyfunc1)
    pyfunc13 = unwrap13
    cfunc13 = njit(pyfunc13)
    pyfunc123 = unwrap123
    cfunc123 = njit(pyfunc123)

    def inputs1():
        yield np.array([1, 1 + 2 * np.pi])
        phase = np.linspace(0, np.pi, num=5)
        phase[3:] += np.pi
        yield phase
        yield np.arange(16).reshape((4, 4))
        yield np.arange(160, step=10).reshape((4, 4))
        yield np.arange(240, step=10).reshape((2, 3, 4))
    for p in inputs1():
        self.assertPreciseEqual(pyfunc1(p), cfunc1(p))
    uneven_seq = np.array([0, 75, 150, 225, 300, 430])
    wrap_uneven = np.mod(uneven_seq, 250)

    def inputs13():
        yield (np.array([1, 1 + 256]), 255)
        yield (np.array([0, 75, 150, 225, 300]), 255)
        yield (np.array([0, 1, 2, -1, 0]), 4)
        yield (np.array([2, 3, 4, 5, 2, 3, 4, 5]), 4)
        yield (wrap_uneven, 250)
    self.assertPreciseEqual(pyfunc(wrap_uneven, axis=-1, period=250), cfunc(wrap_uneven, axis=-1, period=250))
    for p, period in inputs13():
        self.assertPreciseEqual(pyfunc13(p, period=period), cfunc13(p, period=period))

    def inputs123():
        yield (wrap_uneven, 250, 140)
    for p, period, discont in inputs123():
        self.assertPreciseEqual(pyfunc123(p, period=period, discont=discont), cfunc123(p, period=period, discont=discont))