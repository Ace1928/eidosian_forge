from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('asarray', [lambda x: x, lambda x: [np.asarray(a) for a in x], lambda x: [da.asarray(a) for a in x], np.asarray, da.from_array])
@pytest.mark.parametrize('arr, chunks, kwargs', [([[3, 6, 6], [4, 5, 1]], (2, 3), dict(dims=(7, 6), order='C')), ([[3, 6, 6], [4, 5, 1]], (2, 1), dict(dims=(7, 6), order='F')), ([[3, 6, 6], [4, 5, 1]], 1, dict(dims=(4, 6), mode='clip')), ([[3, 6, 6], [4, 5, 1]], (2, 3), dict(dims=(4, 4), mode=('clip', 'wrap'))), ([[3, 6, 6]], (1, 1), dict(dims=7, order='C')), ([[3, 6, 6], [4, 5, 1], [8, 6, 2]], (3, 1), dict(dims=(7, 6, 9), order='C')), (np.arange(6).reshape(3, 2, 1).tolist(), (1, 2, 1), dict(dims=(7, 6, 9), order='C')), ([1, [2, 3]], None, dict(dims=(8, 9))), ([1, [2, 3], [[1, 2], [3, 4], [5, 6], [7, 8]]], None, dict(dims=(8, 9, 10)))])
def test_ravel_multi_index(asarray, arr, chunks, kwargs):
    if any((np.isscalar(x) for x in arr)) and asarray in (np.asarray, da.from_array):
        pytest.skip()
    if asarray is da.from_array:
        arr = np.asarray(arr)
        input = da.from_array(arr, chunks=chunks)
    else:
        arr = input = asarray(arr)
    assert_eq(np.ravel_multi_index(arr, **kwargs), da.ravel_multi_index(input, **kwargs))