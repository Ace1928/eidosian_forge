import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_broadcast_subspace(self):
    a = np.zeros((100, 100))
    v = np.arange(100)[:, None]
    b = np.arange(100)[::-1]
    a[b] = v
    assert_((a[::-1] == v).all())