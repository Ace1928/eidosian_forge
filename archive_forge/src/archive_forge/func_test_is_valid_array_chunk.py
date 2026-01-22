from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
@pytest.mark.parametrize('arr, result', [(WrappedArray(np.arange(4)), False), (da.from_array(np.arange(4)), False), (EncapsulateNDArray(np.arange(4)), True), (np.ma.masked_array(np.arange(4), [True, False, True, False]), True), (np.arange(4), True), (None, True), (0.0, False), (0, False), ('', False)])
def test_is_valid_array_chunk(arr, result):
    """Test is_valid_array_chunk for correctness"""
    assert is_valid_array_chunk(arr) is result