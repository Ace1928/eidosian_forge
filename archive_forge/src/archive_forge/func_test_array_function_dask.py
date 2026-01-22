from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [lambda x: np.append(x, x), lambda x: np.concatenate([x, x, x]), lambda x: np.cov(x, x), lambda x: np.dot(x, x), lambda x: np.dstack((x, x)), lambda x: np.flip(x, axis=0), lambda x: np.hstack((x, x)), lambda x: np.matmul(x, x), lambda x: np.mean(x), lambda x: np.stack([x, x]), lambda x: np.block([x, x]), lambda x: np.sum(x), lambda x: np.var(x), lambda x: np.vstack((x, x)), lambda x: np.linalg.norm(x), lambda x: np.min(x), lambda x: np.amin(x), lambda x: np.round(x), lambda x: np.insert(x, 0, 3, axis=0), lambda x: np.delete(x, 0, axis=0), lambda x: np.select([x < 0.3, x < 0.6, x > 0.7], [x * 2, x, x / 2], default=0.65)])
def test_array_function_dask(func):
    x = np.random.default_rng().random((100, 100))
    y = da.from_array(x, chunks=(50, 50))
    res_x = func(x)
    res_y = func(y)
    assert isinstance(res_y, da.Array)
    assert_eq(res_y, res_x)