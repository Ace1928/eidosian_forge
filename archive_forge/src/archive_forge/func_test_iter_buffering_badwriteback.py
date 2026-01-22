import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering_badwriteback():
    a = np.arange(6).reshape(2, 3, 1)
    b = np.arange(12).reshape(2, 3, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')
    nditer([a, b], ['buffered', 'external_loop'], [['readonly'], ['writeonly']], order='C')
    a = np.arange(1).reshape(1, 1, 1)
    nditer([a, b], ['buffered', 'external_loop', 'reduce_ok'], [['readwrite'], ['writeonly']], order='C')
    a = np.arange(6).reshape(1, 3, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')
    a = np.arange(4).reshape(2, 1, 2)
    assert_raises(ValueError, nditer, [a, b], ['buffered', 'external_loop'], [['readwrite'], ['writeonly']], order='C')