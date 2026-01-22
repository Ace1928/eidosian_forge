import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_diophantine_overflow():
    max_intp = np.iinfo(np.intp).max
    max_int64 = np.iinfo(np.int64).max
    if max_int64 <= max_intp:
        A = (max_int64 // 2, max_int64 // 2 - 10)
        U = (max_int64 // 2, max_int64 // 2 - 10)
        b = 2 * (max_int64 // 2) - 10
        assert_equal(solve_diophantine(A, U, b), (1, 1))