from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
import dask
import dask.array as da
from dask.array.reductions import nannumel, numel
from dask.array.utils import assert_eq
@pytest.mark.skipif(SPARSE_VERSION < parse_version('0.7.0+10'), reason='fixed in https://github.com/pydata/sparse/pull/256')
def test_tensordot():
    x = da.random.random((2, 3, 4), chunks=(1, 2, 2))
    x[x < 0.8] = 0
    y = da.random.random((4, 3, 2), chunks=(2, 2, 1))
    y[y < 0.8] = 0
    xx = x.map_blocks(sparse.COO.from_numpy)
    yy = y.map_blocks(sparse.COO.from_numpy)
    assert_eq(da.tensordot(x, y, axes=(2, 0)), da.tensordot(xx, yy, axes=(2, 0)))
    assert_eq(da.tensordot(x, y, axes=(1, 1)), da.tensordot(xx, yy, axes=(1, 1)))
    assert_eq(da.tensordot(x, y, axes=((1, 2), (1, 0))), da.tensordot(xx, yy, axes=((1, 2), (1, 0))))