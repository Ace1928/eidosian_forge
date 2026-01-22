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
def test_avx_based_ufunc(self):
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    np.random.seed(42)
    for func, prop in avx_ufuncs.items():
        maxulperr = prop[0]
        minval = prop[1]
        maxval = prop[2]
        for size in range(1, 32):
            myfunc = getattr(np, func)
            x_f32 = np.float32(np.random.uniform(low=minval, high=maxval, size=size))
            x_f64 = np.float64(x_f32)
            x_f128 = np.longdouble(x_f32)
            y_true128 = myfunc(x_f128)
            if maxulperr == 0:
                assert_equal(myfunc(x_f32), np.float32(y_true128))
                assert_equal(myfunc(x_f64), np.float64(y_true128))
            else:
                assert_array_max_ulp(myfunc(x_f32), np.float32(y_true128), maxulp=maxulperr)
                assert_array_max_ulp(myfunc(x_f64), np.float64(y_true128), maxulp=maxulperr)
            if size > 1:
                y_true32 = myfunc(x_f32)
                y_true64 = myfunc(x_f64)
                for jj in strides:
                    assert_equal(myfunc(x_f64[::jj]), y_true64[::jj])
                    assert_equal(myfunc(x_f32[::jj]), y_true32[::jj])