import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_broaderrors_indexing(self):
    a = np.zeros((5, 5))
    assert_raises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
    assert_raises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)