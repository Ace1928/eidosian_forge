from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
def broadcast_compat_data(variable: Variable, broadcast_dims: tuple[Hashable, ...], core_dims: tuple[Hashable, ...]) -> Any:
    data = variable.data
    old_dims = variable.dims
    new_dims = broadcast_dims + core_dims
    if new_dims == old_dims:
        return data
    set_old_dims = set(old_dims)
    set_new_dims = set(new_dims)
    unexpected_dims = [d for d in old_dims if d not in set_new_dims]
    if unexpected_dims:
        raise ValueError(f'operand to apply_ufunc encountered unexpected dimensions {unexpected_dims!r} on an input variable: these are core dimensions on other input or output variables')
    old_broadcast_dims = tuple((d for d in broadcast_dims if d in set_old_dims))
    reordered_dims = old_broadcast_dims + core_dims
    if reordered_dims != old_dims:
        order = tuple((old_dims.index(d) for d in reordered_dims))
        data = duck_array_ops.transpose(data, order)
    if new_dims != reordered_dims:
        key_parts: list[slice | None] = []
        for dim in new_dims:
            if dim in set_old_dims:
                key_parts.append(SLICE_NONE)
            elif key_parts:
                key_parts.append(np.newaxis)
        data = data[tuple(key_parts)]
    return data