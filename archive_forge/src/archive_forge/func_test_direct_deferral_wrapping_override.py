from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
def test_direct_deferral_wrapping_override():
    """Directly test Dask deferring to an upcast type and the ability to still wrap it."""
    a = da.from_array(np.arange(4))
    b = WrappedArray(np.arange(4))
    assert a.__add__(b) is NotImplemented
    b.__dask_graph__ = None
    res = a + da.from_array(b)
    assert isinstance(res, da.Array)
    assert_eq(res, 2 * np.arange(4), check_type=False)