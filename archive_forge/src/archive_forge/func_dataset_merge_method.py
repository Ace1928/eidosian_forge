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
def dataset_merge_method(dataset: Dataset, other: CoercibleMapping, overwrite_vars: Hashable | Iterable[Hashable], compat: CompatOptions, join: JoinOptions, fill_value: Any, combine_attrs: CombineAttrsOptions) -> _MergeResult:
    """Guts of the Dataset.merge method."""
    if not isinstance(overwrite_vars, str) and isinstance(overwrite_vars, Iterable):
        overwrite_vars = set(overwrite_vars)
    else:
        overwrite_vars = {overwrite_vars}
    if not overwrite_vars:
        objs = [dataset, other]
        priority_arg = None
    elif overwrite_vars == set(other):
        objs = [dataset, other]
        priority_arg = 1
    else:
        other_overwrite: dict[Hashable, CoercibleValue] = {}
        other_no_overwrite: dict[Hashable, CoercibleValue] = {}
        for k, v in other.items():
            if k in overwrite_vars:
                other_overwrite[k] = v
            else:
                other_no_overwrite[k] = v
        objs = [dataset, other_no_overwrite, other_overwrite]
        priority_arg = 2
    return merge_core(objs, compat, join, priority_arg=priority_arg, fill_value=fill_value, combine_attrs=combine_attrs)