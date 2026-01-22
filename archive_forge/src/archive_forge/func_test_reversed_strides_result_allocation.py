import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_reversed_strides_result_allocation(self):
    a = np.arange(10)[:, None]
    i = np.arange(10)[::-1]
    assert_array_equal(a[i], a[i.copy('C')])
    a = np.arange(20).reshape(-1, 2)