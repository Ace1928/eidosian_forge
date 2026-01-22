from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
@classmethod
def from_variables_maybe_expand(cls, dim: Hashable, current_variables: Mapping[Any, Variable], variables: Mapping[Any, Variable]) -> tuple[PandasMultiIndex, IndexVars]:
    """Create a new multi-index maybe by expanding an existing one with
        new variables as index levels.

        The index and its corresponding coordinates may be created along a new dimension.
        """
    names: list[Hashable] = []
    codes: list[list[int]] = []
    levels: list[list[int]] = []
    level_variables: dict[Any, Variable] = {}
    _check_dim_compat({**current_variables, **variables})
    if len(current_variables) > 1:
        data = cast(PandasMultiIndexingAdapter, next(iter(current_variables.values()))._data)
        current_index = data.array
        names.extend(current_index.names)
        codes.extend(current_index.codes)
        levels.extend(current_index.levels)
        for name in current_index.names:
            level_variables[name] = current_variables[name]
    elif len(current_variables) == 1:
        var = next(iter(current_variables.values()))
        new_var_name = f'{dim}_level_0'
        names.append(new_var_name)
        cat = pd.Categorical(var.values, ordered=True)
        codes.append(cat.codes)
        levels.append(cat.categories)
        level_variables[new_var_name] = var
    for name, var in variables.items():
        names.append(name)
        cat = pd.Categorical(var.values, ordered=True)
        codes.append(cat.codes)
        levels.append(cat.categories)
        level_variables[name] = var
    index = pd.MultiIndex(levels, codes, names=names)
    level_coords_dtype = {k: var.dtype for k, var in level_variables.items()}
    obj = cls(index, dim, level_coords_dtype=level_coords_dtype)
    index_vars = obj.create_variables(level_variables)
    return (obj, index_vars)