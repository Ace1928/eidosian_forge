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
def test_windowing(self):

    def check_window(func):
        np_pyfunc = func
        np_nbfunc = njit(func)
        for M in [0, 1, 5, 12]:
            expected = np_pyfunc(M)
            got = np_nbfunc(M)
            self.assertPreciseEqual(expected, got, prec='double')
        for M in ['a', 1.1, 1j]:
            with self.assertRaises(TypingError) as raises:
                np_nbfunc(1.1)
            self.assertIn('M must be an integer', str(raises.exception))
    check_window(np_bartlett)
    check_window(np_blackman)
    check_window(np_hamming)
    check_window(np_hanning)
    np_pyfunc = np_kaiser
    np_nbfunc = njit(np_kaiser)
    for M in [0, 1, 5, 12]:
        for beta in [0.0, 5.0, 14.0]:
            expected = np_pyfunc(M, beta)
            got = np_nbfunc(M, beta)
            if IS_32BITS or platform.machine() in ['ppc64le', 'aarch64']:
                self.assertPreciseEqual(expected, got, prec='double', ulps=2)
            else:
                self.assertPreciseEqual(expected, got, prec='double', ulps=2)
    for M in ['a', 1.1, 1j]:
        with self.assertRaises(TypingError) as raises:
            np_nbfunc(M, 1.0)
        self.assertIn('M must be an integer', str(raises.exception))
    for beta in ['a', 1j]:
        with self.assertRaises(TypingError) as raises:
            np_nbfunc(5, beta)
        self.assertIn('beta must be an integer or float', str(raises.exception))