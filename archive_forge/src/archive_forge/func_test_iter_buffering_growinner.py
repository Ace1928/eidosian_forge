import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering_growinner():
    a = np.arange(30)
    i = nditer(a, ['buffered', 'growinner', 'external_loop'], buffersize=5)
    assert_equal(i[0].size, a.size)