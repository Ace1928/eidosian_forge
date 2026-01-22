from __future__ import annotations
import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar
import numpy as np
from xarray.core.utils import emit_user_level_warning
from xarray.namedarray.pycompat import is_chunked_array
def guess_chunkmanager(manager: str | ChunkManagerEntrypoint[Any] | None) -> ChunkManagerEntrypoint[Any]:
    """
    Get namespace of chunk-handling methods, guessing from what's available.

    If the name of a specific ChunkManager is given (e.g. "dask"), then use that.
    Else use whatever is installed, defaulting to dask if there are multiple options.
    """
    chunkmanagers = list_chunkmanagers()
    if manager is None:
        if len(chunkmanagers) == 1:
            manager = next(iter(chunkmanagers.keys()))
        else:
            manager = 'dask'
    if isinstance(manager, str):
        if manager not in chunkmanagers:
            raise ValueError(f'unrecognized chunk manager {manager} - must be one of: {list(chunkmanagers)}')
        return chunkmanagers[manager]
    elif isinstance(manager, ChunkManagerEntrypoint):
        return manager
    else:
        raise TypeError(f'manager must be a string or instance of ChunkManagerEntrypoint, but received type {type(manager)}')