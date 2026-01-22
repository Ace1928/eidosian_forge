import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_nontuple_ndindex(self):
    a = np.arange(25).reshape((5, 5))
    assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
    assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
    assert_raises(IndexError, a.__getitem__, [slice(None)])