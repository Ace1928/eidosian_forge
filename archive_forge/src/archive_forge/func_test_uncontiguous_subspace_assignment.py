import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_uncontiguous_subspace_assignment(self):
    a = np.full((3, 4, 2), -1)
    b = np.full((3, 4, 2), -1)
    a[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T
    b[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T.copy()
    assert_equal(a, b)