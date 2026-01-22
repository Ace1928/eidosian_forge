from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_boolean_list_slicing():
    with pytest.raises(IndexError):
        da.asarray(range(2))[[True]]
    with pytest.raises(IndexError):
        da.asarray(range(2))[[False, False, False]]
    x = np.arange(5)
    ind = [True, False, False, False, True]
    assert_eq(da.asarray(x)[ind], x[ind])
    ind = [True]
    assert_eq(da.asarray([0])[ind], np.arange(1)[ind])