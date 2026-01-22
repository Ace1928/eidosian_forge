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

    Mocks the registering of an additional ChunkManagerEntrypoint.

    This preserves the presence of the existing DaskManager, so a test that relies on this and DaskManager both being
    returned from list_chunkmanagers() at once would still work.

    The monkeypatching changes the behavior of list_chunkmanagers when called inside xarray.namedarray.parallelcompat,
    but not when called from this tests file.
    