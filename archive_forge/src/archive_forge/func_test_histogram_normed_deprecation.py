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
def test_histogram_normed_deprecation():
    x = da.arange(10)
    with pytest.raises(ValueError) as info:
        da.histogram(x, bins=[1, 2, 3], normed=True)
    assert 'density' in str(info.value)
    assert 'deprecated' in str(info.value).lower()