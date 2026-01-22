import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_scalar_cast_errors():
    assert_raises(TypeError, nditer, np.float32(2), [], [['readwrite']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, 2.5, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, np.float64(1e+60), [], [['readonly']], casting='safe', op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, np.float32(2), [], [['readonly']], casting='same_kind', op_dtypes=[np.dtype('i4')])