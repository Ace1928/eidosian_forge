from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
@requires_dask
def test_raise_on_mixed_array_types(self, register_dummy_chunkmanager) -> None:
    import dask.array as da
    dummy_arr = DummyChunkedArray([1, 2, 3])
    dask_arr = da.from_array([1, 2, 3], chunks=(1,))
    with pytest.raises(TypeError, match='received multiple types'):
        get_chunked_array_type(*[dask_arr, dummy_arr])