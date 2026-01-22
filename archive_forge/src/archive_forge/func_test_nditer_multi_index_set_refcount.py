import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_nditer_multi_index_set_refcount():
    index = 0
    i = np.nditer(np.array([111, 222, 333, 444]), flags=['multi_index'])
    start_count = sys.getrefcount(index)
    i.multi_index = (index,)
    end_count = sys.getrefcount(index)
    assert_equal(start_count, end_count)