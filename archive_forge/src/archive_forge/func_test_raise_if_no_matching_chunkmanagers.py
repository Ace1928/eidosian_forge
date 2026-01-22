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
def test_raise_if_no_matching_chunkmanagers(self) -> None:
    dummy_arr = DummyChunkedArray([1, 2, 3])
    with pytest.raises(TypeError, match='Could not find a Chunk Manager which recognises'):
        get_chunked_array_type(dummy_arr)