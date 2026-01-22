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
def determine_coords(list_of_mappings: Iterable[DatasetLike]) -> tuple[set[Hashable], set[Hashable]]:
    """Given a list of dicts with xarray object values, identify coordinates.

    Parameters
    ----------
    list_of_mappings : list of dict or list of Dataset
        Of the same form as the arguments to expand_variable_dicts.

    Returns
    -------
    coord_names : set of variable names
    noncoord_names : set of variable names
        All variable found in the input should appear in either the set of
        coordinate or non-coordinate names.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    coord_names: set[Hashable] = set()
    noncoord_names: set[Hashable] = set()
    for mapping in list_of_mappings:
        if isinstance(mapping, Dataset):
            coord_names.update(mapping.coords)
            noncoord_names.update(mapping.data_vars)
        else:
            for name, var in mapping.items():
                if isinstance(var, DataArray):
                    coords = set(var._coords)
                    coords.discard(name)
                    coord_names.update(coords)
    return (coord_names, noncoord_names)