from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Generic, TypeVar
from dask import config
from dask._compatibility import importlib_metadata
from dask.utils import funcname
@lru_cache(maxsize=1)
def detect_entrypoints(entry_point_name):
    return {ep.name: ep for ep in importlib_metadata.entry_points(group=entry_point_name)}