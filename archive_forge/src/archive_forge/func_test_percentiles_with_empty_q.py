from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq, same_keys
@percentile_internal_methods
def test_percentiles_with_empty_q(internal_method):
    x = da.ones(10, chunks=((5, 0, 5),))
    assert_eq(da.percentile(x, [], internal_method=internal_method), np.array([], dtype=x.dtype))