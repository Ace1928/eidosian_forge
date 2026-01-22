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
def assert_unindexed_dim_sizes_equal(self) -> None:
    for dim, sizes in self.unindexed_dim_sizes.items():
        index_size = self.new_indexes.dims.get(dim)
        if index_size is not None:
            sizes.add(index_size)
            add_err_msg = f' (note: an index is found along that dimension with size={index_size!r})'
        else:
            add_err_msg = ''
        if len(sizes) > 1:
            raise ValueError(f'cannot reindex or align along dimension {dim!r} because of conflicting dimension sizes: {sizes!r}' + add_err_msg)