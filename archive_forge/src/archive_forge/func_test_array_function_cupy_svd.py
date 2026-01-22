from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('chunks', [(100, 100), (500, 100)])
def test_array_function_cupy_svd(chunks):
    cupy = pytest.importorskip('cupy')
    x = cupy.random.default_rng().random((500, 100))
    y = da.from_array(x, chunks=chunks, asarray=False)
    u_base, s_base, v_base = da.linalg.svd(y)
    u, s, v = np.linalg.svd(y)
    assert_eq(u, u_base)
    assert_eq(s, s_base)
    assert_eq(v, v_base)