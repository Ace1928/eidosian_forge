import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_indexing_fast_path(self):
    a = np.ones((3, 3))
    idx1 = np.array([[False] * 9])
    assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx1])
    idx2 = np.array([[False] * 8 + [True]])
    assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx2])
    idx3 = np.array([[False] * 10])
    assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx3])
    a = np.ones((1, 1, 2))
    idx = np.array([[[True], [False]]])
    assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 2', lambda: a[idx])