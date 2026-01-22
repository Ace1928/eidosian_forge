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
def interesting_binop_operands(val1, val2, dtype):
    """
    Helper to create "interesting" operands to cover common code paths:
    * scalar inputs
    * only first "values" is an array (e.g. scalar division fast-paths)
    * Longer array (SIMD) placing the value of interest at different positions
    * Oddly strided arrays which may not be SIMD compatible

    It does not attempt to cover unaligned access or mixed dtypes.
    These are normally handled by the casting/buffering machinery.

    This is not a fixture (currently), since I believe a fixture normally
    only yields once?
    """
    fill_value = 1
    arr1 = np.full(10003, dtype=dtype, fill_value=fill_value)
    arr2 = np.full(10003, dtype=dtype, fill_value=fill_value)
    arr1[0] = val1
    arr2[0] = val2
    extractor = lambda res: res
    yield (arr1[0], arr2[0], extractor, 'scalars')
    extractor = lambda res: res
    yield (arr1[0, ...], arr2[0, ...], extractor, 'scalar-arrays')
    arr1[0] = fill_value
    arr2[0] = fill_value
    for pos in [0, 1, 2, 3, 4, 5, -1, -2, -3, -4]:
        arr1[pos] = val1
        arr2[pos] = val2
        extractor = lambda res: res[pos]
        yield (arr1, arr2, extractor, f'off-{pos}')
        yield (arr1, arr2[pos], extractor, f'off-{pos}-with-scalar')
        arr1[pos] = fill_value
        arr2[pos] = fill_value
    for stride in [-1, 113]:
        op1 = arr1[::stride]
        op2 = arr2[::stride]
        op1[10] = val1
        op2[10] = val2
        extractor = lambda res: res[10]
        yield (op1, op2, extractor, f'stride-{stride}')
        op1[10] = fill_value
        op2[10] = fill_value