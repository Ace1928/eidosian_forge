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
@pytest.mark.parametrize('shape', [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)])
@pytest.mark.parametrize('reps', [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)])
def test_tile_np_kroncompare_examples(shape, reps):
    x = np.random.random(shape)
    d = da.asarray(x)
    assert_eq(np.tile(x, reps), da.tile(d, reps))