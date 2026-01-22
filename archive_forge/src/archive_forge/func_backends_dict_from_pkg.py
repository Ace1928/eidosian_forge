from __future__ import annotations
import functools
import inspect
import itertools
import sys
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.utils import module_available
def backends_dict_from_pkg(entrypoints: list[EntryPoint]) -> dict[str, type[BackendEntrypoint]]:
    backend_entrypoints = {}
    for entrypoint in entrypoints:
        name = entrypoint.name
        try:
            backend = entrypoint.load()
            backend_entrypoints[name] = backend
        except Exception as ex:
            warnings.warn(f'Engine {name!r} loading failed:\n{ex}', RuntimeWarning)
    return backend_entrypoints