import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_errors():
    a = arange(6)
    assert_raises(TypeError, nditer, [a, None], [], [['writeonly'], ['writeonly', 'allocate']])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['allocate', 'readonly']])
    assert_raises(ValueError, nditer, [a, None], ['buffered'], ['allocate', 'readwrite'])
    assert_raises(TypeError, nditer, [None, None], [], [['writeonly', 'allocate'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')])
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, np.newaxis, 1]])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, 3, 1]])
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['writeonly', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, 2, 1, 0]])
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], ['reduce_ok'], [['readonly'], ['readwrite', 'allocate']], op_dtypes=[None, np.dtype('f4')], op_axes=[None, [0, np.newaxis, 2]])