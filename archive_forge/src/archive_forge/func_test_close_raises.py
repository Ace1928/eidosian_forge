import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_close_raises():
    it = np.nditer(np.arange(3))
    assert_equal(next(it), 0)
    it.close()
    assert_raises(StopIteration, next, it)
    assert_raises(ValueError, getattr, it, 'operands')