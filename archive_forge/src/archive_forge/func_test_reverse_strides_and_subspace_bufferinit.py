import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_reverse_strides_and_subspace_bufferinit(self):
    a = np.ones(5)
    b = np.zeros(5, dtype=np.intp)[::-1]
    c = np.arange(5)[::-1]
    a[b] = c
    assert_equal(a[0], 0)
    a = np.ones((5, 2))
    c = np.arange(10).reshape(5, 2)[::-1]
    a[b, :] = c
    assert_equal(a[0], [0, 1])