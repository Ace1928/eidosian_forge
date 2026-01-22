import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_indexing_onedim(self):
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([True], dtype=bool)
    assert_equal(a[b], a)
    a[b] = 1.0
    assert_equal(a, [[1.0, 1.0, 1.0]])