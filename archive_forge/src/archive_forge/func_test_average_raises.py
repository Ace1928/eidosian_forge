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
def test_average_raises():
    d_a = da.arange(11, chunks=2)
    with pytest.raises(TypeError):
        da.average(d_a, weights=[1, 2, 3])
    with pytest.warns(RuntimeWarning):
        da.average(d_a, weights=da.zeros_like(d_a)).compute()