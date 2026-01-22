from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.reshape import contract_tuple, expand_tuple, reshape_rechunk
from dask.array.utils import assert_eq
def test_reshape_unknown_sizes():
    a = np.random.random((10, 6, 6))
    A = da.from_array(a, chunks=(5, 2, 3))
    a2 = a.reshape((60, -1))
    A2 = A.reshape((60, -1))
    assert A2.shape == (60, 6)
    assert_eq(A2, a2)
    with pytest.raises(ValueError):
        a.reshape((60, -1, -1))
    with pytest.raises(ValueError):
        A.reshape((60, -1, -1))