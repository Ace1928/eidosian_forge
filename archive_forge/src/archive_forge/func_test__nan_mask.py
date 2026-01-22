import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
@pytest.mark.parametrize('arr, expected', [(np.array([np.nan, 5.0, np.nan, np.inf]), np.array([False, True, False, True])), (np.array([1, 5, 7, 9], dtype=np.int64), True), (np.array([False, True, False, True]), True), (np.array([[np.nan, 5.0], [np.nan, np.inf]], dtype=np.complex64), np.array([[False, True], [False, True]]))])
def test__nan_mask(arr, expected):
    for out in [None, np.empty(arr.shape, dtype=np.bool_)]:
        actual = _nan_mask(arr, out=out)
        assert_equal(actual, expected)
        if type(expected) is not np.ndarray:
            assert actual is True