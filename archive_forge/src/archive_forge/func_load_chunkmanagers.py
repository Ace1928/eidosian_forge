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
def load_chunkmanagers(entrypoints: Sequence[EntryPoint]) -> dict[str, ChunkManagerEntrypoint[Any]]:
    """Load entrypoints and instantiate chunkmanagers only once."""
    loaded_entrypoints = {}
    for entrypoint in entrypoints:
        try:
            loaded_entrypoints[entrypoint.name] = entrypoint.load()
        except ModuleNotFoundError as e:
            emit_user_level_warning(f'Failed to load chunk manager entrypoint {entrypoint.name} due to {e}. Skipping.')
            pass
    available_chunkmanagers = {name: chunkmanager() for name, chunkmanager in loaded_entrypoints.items() if chunkmanager.available}
    return available_chunkmanagers