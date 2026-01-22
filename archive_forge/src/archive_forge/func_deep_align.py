from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def deep_align(objects: Iterable[Any], join: JoinOptions='inner', copy: bool=True, indexes=None, exclude: str | Iterable[Hashable]=frozenset(), raise_on_invalid: bool=True, fill_value=dtypes.NA) -> list[Any]:
    """Align objects for merging, recursing into dictionary values.

    This function is not public API.
    """
    from xarray.core.coordinates import Coordinates
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    if indexes is None:
        indexes = {}

    def is_alignable(obj):
        return isinstance(obj, (Coordinates, DataArray, Dataset))
    positions: list[int] = []
    keys: list[type[object] | Hashable] = []
    out: list[Any] = []
    targets: list[Alignable] = []
    no_key: Final = object()
    not_replaced: Final = object()
    for position, variables in enumerate(objects):
        if is_alignable(variables):
            positions.append(position)
            keys.append(no_key)
            targets.append(variables)
            out.append(not_replaced)
        elif is_dict_like(variables):
            current_out = {}
            for k, v in variables.items():
                if is_alignable(v) and k not in indexes:
                    positions.append(position)
                    keys.append(k)
                    targets.append(v)
                    current_out[k] = not_replaced
                else:
                    current_out[k] = v
            out.append(current_out)
        elif raise_on_invalid:
            raise ValueError(f'object to align is neither an xarray.Dataset, an xarray.DataArray nor a dictionary: {variables!r}')
        else:
            out.append(variables)
    aligned = align(*targets, join=join, copy=copy, indexes=indexes, exclude=exclude, fill_value=fill_value)
    for position, key, aligned_obj in zip(positions, keys, aligned):
        if key is no_key:
            out[position] = aligned_obj
        else:
            out[position][key] = aligned_obj
    return out