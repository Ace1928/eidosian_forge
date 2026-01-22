import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_index_cast_assign(self):
    shape = (8, 63)
    bool_index = np.zeros(shape).astype(bool)
    bool_index[0, 1] = True
    zero_array = np.zeros(shape)
    zero_array[bool_index] = np.array([1])
    assert_equal(zero_array[0, 1], 1)
    assert_warns(np.ComplexWarning, zero_array.__setitem__, ([0], [1]), np.array([2 + 1j]))
    assert_equal(zero_array[0, 1], 2)
    assert_warns(np.ComplexWarning, zero_array.__setitem__, bool_index, np.array([1j]))
    assert_equal(zero_array[0, 1], 0)