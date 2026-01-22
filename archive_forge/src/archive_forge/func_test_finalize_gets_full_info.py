import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_finalize_gets_full_info(self):

    class SubClass(np.ndarray):

        def __array_finalize__(self, old):
            self.finalize_status = np.array(self)
            self.old = old
    s = np.arange(10).view(SubClass)
    new_s = s[:3]
    assert_array_equal(new_s.finalize_status, new_s)
    assert_array_equal(new_s.old, s)
    new_s = s[[0, 1, 2, 3]]
    assert_array_equal(new_s.finalize_status, new_s)
    assert_array_equal(new_s.old, s)
    new_s = s[s > 0]
    assert_array_equal(new_s.finalize_status, new_s)
    assert_array_equal(new_s.old, s)