from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq, same_keys
@percentile_internal_methods
@pytest.mark.parametrize('q', [5, 5.0, np.int64(5), np.float64(5)])
def test_percentiles_with_scaler_percentile(internal_method, q):
    d = da.ones((16,), chunks=(4,))
    assert_eq(da.percentile(d, q, internal_method=internal_method), np.array([1], dtype=d.dtype))