from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
@pytest.mark.parametrize('arr', [da.from_array([1, 2]), np.asarray([1, 2])])
def test_delegation_unknown_scalar(arr):
    s = UnknownScalar()
    assert arr * s == 42
    assert s * arr == 42
    with pytest.raises(TypeError, match="operand 'UnknownScalar' does not support ufuncs"):
        np.multiply(s, arr)