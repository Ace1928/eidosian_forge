from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [np.equal, np.matmul, np.dot, lambda x, y: np.stack([x, y])])
@pytest.mark.parametrize('arr_upcast, arr_downcast', [(WrappedArray(np.random.default_rng().random((10, 10))), da.random.default_rng().random((10, 10), chunks=(5, 5))), (da.random.default_rng().random((10, 10), chunks=(5, 5)), EncapsulateNDArray(np.random.default_rng().random((10, 10)))), (WrappedArray(np.random.default_rng().random((10, 10))), EncapsulateNDArray(np.random.default_rng().random((10, 10))))])
def test_binary_function_type_precedence(func, arr_upcast, arr_downcast):
    """Test proper dispatch on binary NumPy functions"""
    assert type(func(arr_upcast, arr_downcast)) == type(func(arr_downcast, arr_upcast)) == type(arr_upcast)