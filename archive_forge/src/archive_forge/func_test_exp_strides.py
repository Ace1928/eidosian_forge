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
def test_exp_strides(self):
    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = np.arange(2, 100)
    for ii in sizes:
        x_f64 = np.float64(np.random.uniform(low=0.01, high=709.1, size=ii))
        y_true = np.exp(x_f64)
        for jj in strides:
            assert_array_almost_equal_nulp(np.exp(x_f64[::jj]), y_true[::jj], nulp=2)