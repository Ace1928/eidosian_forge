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
def test_trim_zeros_numpy(self):
    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    values = [a, b, c]
    slc = np.s_[2:-1]
    for arr in values:
        res = np_trim_zeros(arr)
        self.assertPreciseEqual(res, arr[slc])
    slc = np.s_[:-1]
    for arr in values:
        res = np_trim_zeros(arr, trim='b')
        self.assertPreciseEqual(res, arr[slc])
    slc = np.s_[2:]
    for arr in values:
        res = np_trim_zeros(arr, trim='F')
        self.assertPreciseEqual(res, arr[slc])
    for _arr in values:
        arr = np.zeros_like(_arr, dtype=_arr.dtype)
        res1 = np_trim_zeros(arr, trim='B')
        assert len(res1) == 0
        res2 = np_trim_zeros(arr, trim='f')
        assert len(res2) == 0
    arr = np.zeros(0)
    res = np_trim_zeros(arr)
    self.assertPreciseEqual(arr, res)
    for arr in [np.array([0, 2 ** 62, 0]), np.array([0, 2 ** 63, 0]), np.array([0, 2 ** 64, 0])]:
        slc = np.s_[1:2]
        res = np_trim_zeros(arr)
        self.assertPreciseEqual(res, arr[slc])
    arr = np.array([None, 1, None])
    res = np_trim_zeros(arr)
    self.assertPreciseEqual(arr, res)
    res = np_trim_zeros(a.tolist())
    assert isinstance(res, list)