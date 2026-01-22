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
def test_slicing_integer_no_warnings():
    X = da.random.default_rng().random(size=(100, 2), chunks=(2, 2))
    idx = np.array([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as record:
        X[idx].compute()
    assert not record