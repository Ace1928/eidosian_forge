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
@pytest.mark.skip(reason='really long test')
def test_slicing_exhaustively():
    x = np.random.default_rng().random(6, 7, 8)
    a = da.from_array(x, chunks=(3, 3, 3))
    I = ReturnItem()
    indexers = [0, -2, I[:], I[:5], [0, 1], [0, 1, 2], [4, 2], I[::-1], None, I[:0], []]
    for i in indexers:
        assert_eq(x[i], a[i])
        for j in indexers:
            assert_eq(x[i][:, j], a[i][:, j])
            assert_eq(x[:, i][j], a[:, i][j])
            for k in indexers:
                assert_eq(x[..., i][:, j][k], a[..., i][:, j][k])
    first_indexers = [I[:], I[:5], np.arange(5), [3, 1, 4, 5, 0], np.arange(6) < 6]
    second_indexers = [0, -1, 3, I[:], I[:3], I[2:-1], [2, 4], [], I[:0]]
    for i in first_indexers:
        for j in second_indexers:
            assert_eq(x[i][j], a[i][j])