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
def test_arange_float_step():
    darr = da.arange(2.0, 13.0, 0.3, chunks=4)
    nparr = np.arange(2.0, 13.0, 0.3)
    assert_eq(darr, nparr)
    darr = da.arange(7.7, 1.5, -0.8, chunks=3)
    nparr = np.arange(7.7, 1.5, -0.8)
    assert_eq(darr, nparr)
    darr = da.arange(0, 1, 0.01, chunks=20)
    nparr = np.arange(0, 1, 0.01)
    assert_eq(darr, nparr)
    darr = da.arange(0, 1, 0.03, chunks=20)
    nparr = np.arange(0, 1, 0.03)
    assert_eq(darr, nparr)