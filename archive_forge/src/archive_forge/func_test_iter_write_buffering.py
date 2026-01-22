import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_write_buffering():
    a = np.arange(24).reshape(2, 3, 4).T.newbyteorder().byteswap()
    i = nditer(a, ['buffered'], [['readwrite', 'nbo', 'aligned']], casting='equiv', order='C', buffersize=16)
    x = 0
    with i:
        while not i.finished:
            i[0] = x
            x += 1
            i.iternext()
    assert_equal(a.ravel(order='C'), np.arange(24))