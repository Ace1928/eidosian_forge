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
def test_bogus_entrypoint() -> None:
    entry_point = EntryPoint('bogus', 'xarray.bogus.doesnotwork', 'xarray.chunkmanagers')
    with pytest.warns(UserWarning, match='Failed to load chunk manager'):
        assert len(load_chunkmanagers([entry_point])) == 0