from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('reps', [2, (2, 2), (1, 2), (2, 1), (2, 3, 4, 0)])
def test_tile_basic(reps):
    a = da.asarray([0, 1, 2])
    b = [[1, 2], [3, 4]]
    assert_eq(np.tile(a.compute(), reps), da.tile(a, reps))
    assert_eq(np.tile(b, reps), da.tile(b, reps))