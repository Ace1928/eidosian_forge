from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_create_with_auto_dimensions():
    with dask.config.set({'array.chunk-size': '128MiB'}):
        x = da.random.random((10000, 10000), chunks=(-1, 'auto'))
        assert x.chunks == ((10000,), (1677,) * 5 + (1615,))
        y = da.random.random((10000, 10000), chunks='auto')
        assert y.chunks == ((4096, 4096, 1808),) * 2