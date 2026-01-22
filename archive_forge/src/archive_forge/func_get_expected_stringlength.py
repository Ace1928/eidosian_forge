import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def get_expected_stringlength(dtype):
    """Returns the string length when casting the basic dtypes to strings.
    """
    if dtype == np.bool_:
        return 5
    if dtype.kind in 'iu':
        if dtype.itemsize == 1:
            length = 3
        elif dtype.itemsize == 2:
            length = 5
        elif dtype.itemsize == 4:
            length = 10
        elif dtype.itemsize == 8:
            length = 20
        else:
            raise AssertionError(f'did not find expected length for {dtype}')
        if dtype.kind == 'i':
            length += 1
        return length
    if dtype.char == 'g':
        return 48
    elif dtype.char == 'G':
        return 48 * 2
    elif dtype.kind == 'f':
        return 32
    elif dtype.kind == 'c':
        return 32 * 2
    raise AssertionError(f'did not find expected length for {dtype}')