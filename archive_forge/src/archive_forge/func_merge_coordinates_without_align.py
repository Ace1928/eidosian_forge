from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.alignment import deep_align
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import (
from xarray.core.utils import Frozen, compat_dict_union, dict_equiv, equivalent
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def merge_coordinates_without_align(objects: list[Coordinates], prioritized: Mapping[Any, MergeElement] | None=None, exclude_dims: Set=frozenset(), combine_attrs: CombineAttrsOptions='override') -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge variables/indexes from coordinates without automatic alignments.

    This function is used for merging coordinate from pre-existing xarray
    objects.
    """
    collected = collect_from_coordinates(objects)
    if exclude_dims:
        filtered: dict[Hashable, list[MergeElement]] = {}
        for name, elements in collected.items():
            new_elements = [(variable, index) for variable, index in elements if exclude_dims.isdisjoint(variable.dims)]
            if new_elements:
                filtered[name] = new_elements
    else:
        filtered = collected
    merged_coords, merged_indexes = merge_collected(filtered, prioritized, combine_attrs=combine_attrs)
    merged_indexes = filter_indexes_from_coords(merged_indexes, set(merged_coords))
    return (merged_coords, merged_indexes)