import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_integer_split_2D_cols(self):
    a = np.array([np.arange(10), np.arange(10)])
    res = array_split(a, 3, axis=-1)
    desired = [np.array([np.arange(4), np.arange(4)]), np.array([np.arange(4, 7), np.arange(4, 7)]), np.array([np.arange(7, 10), np.arange(7, 10)])]
    compare_results(res, desired)