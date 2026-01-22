from __future__ import annotations
import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast
import numpy as np
from packaging.version import Version
from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike
def is_duck_array(value: Any) -> TypeGuard[duckarray[Any, Any]]:
    if isinstance(value, np.ndarray):
        return True
    return hasattr(value, 'ndim') and hasattr(value, 'shape') and hasattr(value, 'dtype') and (hasattr(value, '__array_function__') and hasattr(value, '__array_ufunc__') or hasattr(value, '__array_namespace__'))