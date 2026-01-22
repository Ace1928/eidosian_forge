import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_assignment_needs_api(self):
    arr = np.zeros(1000)
    indx = np.zeros(1000, dtype=bool)
    indx[:100] = True
    arr[indx] = np.ones(100, dtype=object)
    expected = np.zeros(1000)
    expected[:100] = 1
    assert_array_equal(arr, expected)