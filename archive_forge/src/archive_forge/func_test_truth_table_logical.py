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
def test_truth_table_logical(self):
    input1 = [0, 0, 3, 2]
    input2 = [0, 4, 0, 2]
    typecodes = np.typecodes['AllFloat'] + np.typecodes['AllInteger'] + '?'
    for dtype in map(np.dtype, typecodes):
        arg1 = np.asarray(input1, dtype=dtype)
        arg2 = np.asarray(input2, dtype=dtype)
        out = [False, True, True, True]
        for func in (np.logical_or, np.maximum):
            assert_equal(func(arg1, arg2).astype(bool), out)
        out = [False, False, False, True]
        for func in (np.logical_and, np.minimum):
            assert_equal(func(arg1, arg2).astype(bool), out)
        out = [False, True, True, False]
        for func in (np.logical_xor, np.not_equal):
            assert_equal(func(arg1, arg2).astype(bool), out)