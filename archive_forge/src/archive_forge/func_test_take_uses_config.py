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
def test_take_uses_config():
    with dask.config.set({'array.slicing.split-large-chunks': True}):
        chunks = ((1, 1, 1, 1), (500,), (500,))
        index = np.array([0, 1] + [2] * 101 + [3])
        itemsize = 8
        with config.set({'array.chunk-size': '10GB'}):
            chunks2, dsk = take('a', 'b', chunks, index, itemsize)
        assert chunks2 == ((1, 1, 101, 1), (500,), (500,))
        assert len(dsk) == 4