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
@pytest.mark.parametrize('stride', [-5, -4, -3, -2, -1, 1, 2, 4, 5, 6, 7, 8, 9, 10])
def test_fpclass(self, stride):
    arr_f64 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.2251e-308], dtype='d')
    arr_f32 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 1.4013e-45, -1.4013e-45], dtype='f')
    nan = np.array([True, True, False, False, False, False, False, False, False, False])
    inf = np.array([False, False, True, True, False, False, False, False, False, False])
    sign = np.array([False, True, False, True, True, False, True, False, False, True])
    finite = np.array([False, False, False, False, True, True, True, True, True, True])
    assert_equal(np.isnan(arr_f32[::stride]), nan[::stride])
    assert_equal(np.isnan(arr_f64[::stride]), nan[::stride])
    assert_equal(np.isinf(arr_f32[::stride]), inf[::stride])
    assert_equal(np.isinf(arr_f64[::stride]), inf[::stride])
    assert_equal(np.signbit(arr_f32[::stride]), sign[::stride])
    assert_equal(np.signbit(arr_f64[::stride]), sign[::stride])
    assert_equal(np.isfinite(arr_f32[::stride]), finite[::stride])
    assert_equal(np.isfinite(arr_f64[::stride]), finite[::stride])