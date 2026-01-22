import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_types_promotion():
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f4'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('u4'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), None], [], [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'))