import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_sincos_float32(self):
    np.random.seed(42)
    N = 1000000
    M = np.int_(N / 20)
    index = np.random.randint(low=0, high=N, size=M)
    x_f32 = np.float32(np.random.uniform(low=-100.0, high=100.0, size=N))
    if not _glibc_older_than('2.17'):
        x_f32[index] = np.float32(100000000000.0 * np.random.rand(M))
    x_f64 = np.float64(x_f32)
    assert_array_max_ulp(np.sin(x_f32), np.float32(np.sin(x_f64)), maxulp=2)
    assert_array_max_ulp(np.cos(x_f32), np.float32(np.cos(x_f64)), maxulp=2)
    tx_f32 = x_f32.copy()
    assert_array_max_ulp(np.sin(x_f32, out=x_f32), np.float32(np.sin(x_f64)), maxulp=2)
    assert_array_max_ulp(np.cos(tx_f32, out=tx_f32), np.float32(np.cos(x_f64)), maxulp=2)