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
@pytest.mark.xfail(reason='broadcasting in da.select() not implemented yet')
def test_select_broadcasting():
    conditions = [np.array(True), np.array([False, True, False])]
    choices = [1, np.arange(12).reshape(4, 3)]
    d_conditions = da.from_array(conditions)
    d_choices = da.from_array(choices)
    assert_eq(np.select(conditions, choices), da.select(d_conditions, d_choices))
    assert_eq(np.select([True], [0], default=[0]), da.select([True], [0], default=[0]))