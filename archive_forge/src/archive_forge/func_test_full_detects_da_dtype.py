from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_full_detects_da_dtype():
    x = da.from_array(100)
    with pytest.warns(FutureWarning, match='not implemented by Dask array') as record:
        a = da.full(shape=(3, 3), fill_value=x)
        assert a.dtype == x.dtype
        assert_eq(a, np.full(shape=(3, 3), fill_value=100))
    assert len(record) == 1