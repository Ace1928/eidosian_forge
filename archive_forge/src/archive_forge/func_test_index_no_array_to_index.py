import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_index_no_array_to_index(self):
    a = np.array([[[1]]])
    assert_raises(TypeError, lambda: a[a:a:a])