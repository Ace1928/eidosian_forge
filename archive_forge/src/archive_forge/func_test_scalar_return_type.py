import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_scalar_return_type(self):
    a = np.zeros((), [('a', 'f8')])
    assert_(isinstance(a['a'], np.ndarray))
    assert_(isinstance(a[['a']], np.ndarray))