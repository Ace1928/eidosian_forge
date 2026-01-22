from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_full_none_dtype():
    a = da.full(shape=(3, 3), fill_value=100, dtype=None)
    assert_eq(a, np.full(shape=(3, 3), fill_value=100, dtype=None))