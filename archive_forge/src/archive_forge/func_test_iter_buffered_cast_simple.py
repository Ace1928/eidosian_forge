import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffered_cast_simple():
    a = np.arange(10, dtype='f4')
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('f8')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='f4'))