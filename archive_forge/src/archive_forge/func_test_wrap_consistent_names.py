from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_wrap_consistent_names():
    assert sorted(ones(10, dtype='i4', chunks=(4,)).dask) == sorted(ones(10, dtype='i4', chunks=(4,)).dask)
    assert sorted(ones(10, dtype='i4', chunks=(4,)).dask) != sorted(ones(10, chunks=(4,)).dask)
    assert sorted(da.full((3, 3), 100, chunks=(2, 2), dtype='f8').dask) == sorted(da.full((3, 3), 100, chunks=(2, 2), dtype='f8').dask)
    assert sorted(da.full((3, 3), 100, chunks=(2, 2), dtype='i2').dask) != sorted(da.full((3, 3), 100, chunks=(2, 2)).dask)