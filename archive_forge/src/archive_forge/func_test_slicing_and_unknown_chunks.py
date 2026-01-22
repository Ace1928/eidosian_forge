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
def test_slicing_and_unknown_chunks():
    a = da.ones((10, 5), chunks=5)
    a._chunks = ((np.nan, np.nan), (5,))
    with pytest.raises(ValueError, match='Array chunk size or shape is unknown'):
        a[[0, 5]].compute()