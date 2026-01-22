import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_index_is_larger(self):
    a = np.zeros((5, 5))
    a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]
    assert_((a[:3, :3] == [2, 3, 4]).all())