from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('chunks', [((5, 5, 2), (5, 5, 2)), ((3, 3, 3, 3), (11, 1))])
def test_depth_greater_than_smallest_chunk_combines_chunks(chunks):
    a = np.arange(144).reshape(12, 12)
    darr = da.from_array(a, chunks=chunks)
    depth = {0: 4, 1: 2}
    output = overlap(darr, depth=depth, boundary=1)
    assert all((c >= depth[0] * 2 for c in output.chunks[0]))
    assert all((c >= depth[1] * 2 for c in output.chunks[1]))