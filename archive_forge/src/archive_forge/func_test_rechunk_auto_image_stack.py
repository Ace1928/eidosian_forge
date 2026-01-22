from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
@pytest.mark.parametrize('n', [100, 1000])
def test_rechunk_auto_image_stack(n):
    with dask.config.set({'array.chunk-size': '10MiB'}):
        x = da.ones((n, 1000, 1000), chunks=(1, 1000, 1000), dtype='uint8')
        y = x.rechunk('auto')
        assert y.chunks == ((10,) * (n // 10), (1000,), (1000,))
        assert y.rechunk('auto').chunks == y.chunks
    with dask.config.set({'array.chunk-size': '7MiB'}):
        z = x.rechunk('auto')
        if n == 100:
            assert z.chunks == ((7,) * 14 + (2,), (1000,), (1000,))
        else:
            assert z.chunks == ((7,) * 142 + (6,), (1000,), (1000,))
    with dask.config.set({'array.chunk-size': '1MiB'}):
        x = da.ones((n, 1000, 1000), chunks=(1, 1000, 1000), dtype='float64')
        z = x.rechunk('auto')
        assert z.chunks == ((1,) * n, (362, 362, 276), (362, 362, 276))