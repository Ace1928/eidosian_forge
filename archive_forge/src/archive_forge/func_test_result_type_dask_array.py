from __future__ import annotations
import numpy as np
import pytest
from xarray.core import dtypes
def test_result_type_dask_array() -> None:
    da = pytest.importorskip('dask.array')
    dask = pytest.importorskip('dask')

    def error():
        raise RuntimeError
    array = da.from_delayed(dask.delayed(error)(), (), np.float64)
    with pytest.raises(RuntimeError):
        array.compute()
    actual = dtypes.result_type(array)
    assert actual == np.float64
    actual = dtypes.result_type(array, np.array([0.5, 1.0], dtype=np.float32))
    assert actual == np.float64