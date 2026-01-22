from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [np.fft.fft, np.fft.fft2])
def test_array_function_fft(func):
    x = np.random.default_rng().random((100, 100))
    y = da.from_array(x, chunks=(100, 100))
    res_x = func(x)
    res_y = func(y)
    if func.__module__ != 'mkl_fft._numpy_fft':
        assert isinstance(res_y, da.Array)
    assert_eq(res_y, res_x)