from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_array_of_different_depths(self):
    arr = np.zeros((3, 2))
    mismatch_first_dim = np.zeros((1, 2))
    mismatch_second_dim = np.zeros((3, 3))
    dtype, shape = _discover_array_parameters([arr, mismatch_second_dim], dtype=np.dtype('O'))
    assert shape == (2, 3)
    dtype, shape = _discover_array_parameters([arr, mismatch_first_dim], dtype=np.dtype('O'))
    assert shape == (2,)
    res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype('O'))
    assert res[0] is arr
    assert res[1] is mismatch_first_dim