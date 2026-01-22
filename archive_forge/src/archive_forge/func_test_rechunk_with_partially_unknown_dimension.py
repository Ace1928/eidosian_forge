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
@pytest.mark.parametrize('x, chunks', [(da.ones(shape=(50, 10), chunks=(25, 10)), (None, 5)), (da.ones(shape=(50, 10), chunks=(25, 10)), {1: 5}), (da.ones(shape=(50, 10), chunks=(25, 10)), (None, (5, 5))), (da.ones(shape=(1000, 10), chunks=(5, 10)), (None, 5)), (da.ones(shape=(1000, 10), chunks=(5, 10)), {1: 5}), (da.ones(shape=(1000, 10), chunks=(5, 10)), (None, (5, 5))), (da.ones(shape=(10, 10), chunks=(10, 10)), (None, 5)), (da.ones(shape=(10, 10), chunks=(10, 10)), {1: 5}), (da.ones(shape=(10, 10), chunks=(10, 10)), (None, (5, 5))), (da.ones(shape=(10, 10), chunks=(10, 2)), (None, 5)), (da.ones(shape=(10, 10), chunks=(10, 2)), {1: 5}), (da.ones(shape=(10, 10), chunks=(10, 2)), (None, (5, 5)))])
def test_rechunk_with_partially_unknown_dimension(x, chunks):
    dd = pytest.importorskip('dask.dataframe')
    y = dd.from_array(x).values
    z = da.concatenate([x, y])
    xx = da.concatenate([x, x])
    result = z.rechunk(chunks)
    expected = xx.rechunk(chunks)
    assert_chunks_match(result.chunks, expected.chunks)
    assert_eq(result, expected)