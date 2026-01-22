import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_cast_equivalence(self):
    a = np.arange(5)
    b = a.copy()
    a[:3] = np.array(['2', '-3', '-1'])
    b[[0, 2, 1]] = np.array(['2', '-1', '-3'])
    assert_array_equal(a, b)
    b = np.arange(5)[None, :]
    b[[0], :3] = np.array([['2', '-3', '-1']])
    assert_array_equal(a, b[0])