from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_pathological_self_containing(self):
    l = []
    l.append(l)
    arr = np.array([l, l, l], dtype=object)
    assert arr.shape == (3,) + (1,) * (np.MAXDIMS - 1)
    arr = np.array([l, [None], l], dtype=object)
    assert arr.shape == (3, 1)