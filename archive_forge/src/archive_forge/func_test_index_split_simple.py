import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_index_split_simple(self):
    a = np.arange(10)
    indices = [1, 5, 7]
    res = array_split(a, indices, axis=-1)
    desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7), np.arange(7, 10)]
    compare_results(res, desired)