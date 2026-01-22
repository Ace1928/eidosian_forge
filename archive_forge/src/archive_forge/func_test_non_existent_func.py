from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
def test_non_existent_func():
    x = da.from_array(np.array([1, 2, 4, 3]), chunks=(2,))
    with pytest.warns(FutureWarning, match='The `numpy.sort` function is not implemented by Dask'):
        assert list(np.sort(x)) == [1, 2, 3, 4]