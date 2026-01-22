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
def test_select_exception(self):
    np_nbfunc = njit(np_select)
    x = np.arange(10)
    self.disable_leak_check()
    for condlist, choicelist, default, expected_error, expected_text in [([np.array(True), np.array([False, True, False])], [np.array(1), np.arange(12).reshape(4, 3)], 0, TypingError, 'condlist arrays must be of at least dimension 1'), ([np.array(True), np.array(False)], [np.array([1]), np.array([2])], 0, TypingError, 'condlist and choicelist elements must have the same number of dimensions'), ([np.array([True]), np.array([False])], [np.array([[1]]), np.array([[2]])], 0, TypingError, 'condlist and choicelist elements must have the same number of dimensions'), ([np.array(True), np.array(False)], [np.array(1), np.array(2)], 0, TypingError, 'condlist arrays must be of at least dimension 1'), (np.isnan(np.array([1, 2, 3, np.nan, 5, 7])), np.array([1, 2, 3, np.nan, 5, 7]), 0, TypingError, 'condlist must be a List or a Tuple'), ([True], [0], [0], TypingError, 'default must be a scalar'), ([(x < 3).astype(int), (x > 5).astype(int)], [x, x ** 2], 0, TypingError, 'condlist arrays must contain booleans'), ([x > 9, x > 8, x > 7, x > 6], [x, x ** 2, x], 0, ValueError, 'list of cases must be same length as list of conditions'), ([(False,)] * 100, [np.array([1])] * 100, 0, TypingError, 'items of condlist must be arrays'), ([np.array([False])] * 100, [(1,)] * 100, 0, TypingError, 'items of choicelist must be arrays')]:
        with self.assertRaises(expected_error) as e:
            np_nbfunc(condlist, choicelist, default)
        self.assertIn(expected_text, str(e.exception))