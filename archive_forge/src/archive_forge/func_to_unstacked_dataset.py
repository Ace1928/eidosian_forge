from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import (
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import (
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs
from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims
def to_unstacked_dataset(self, dim: Hashable, level: int | Hashable=0) -> Dataset:
    """Unstack DataArray expanding to Dataset along a given level of a
        stacked coordinate.

        This is the inverse operation of Dataset.to_stacked_array.

        Parameters
        ----------
        dim : Hashable
            Name of existing dimension to unstack
        level : int or Hashable, default: 0
            The MultiIndex level to expand to a dataset along. Can either be
            the integer index of the level or its name.

        Returns
        -------
        unstacked: Dataset

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],
        ... )
        >>> data = xr.Dataset({"a": arr, "b": arr.isel(y=0)})
        >>> data
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
          * y        (y) int64 24B 0 1 2
        Data variables:
            a        (x, y) int64 48B 0 1 2 3 4 5
            b        (x) int64 16B 0 3
        >>> stacked = data.to_stacked_array("z", ["x"])
        >>> stacked.indexes["z"]
        MultiIndex([('a',   0),
                    ('a',   1),
                    ('a',   2),
                    ('b', nan)],
                   name='z')
        >>> roundtripped = stacked.to_unstacked_dataset(dim="z")
        >>> data.identical(roundtripped)
        True

        See Also
        --------
        Dataset.to_stacked_array
        """
    idx = self._indexes[dim].to_pandas_index()
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError(f"'{dim}' is not a stacked coordinate")
    level_number = idx._get_level_number(level)
    variables = idx.levels[level_number]
    variable_dim = idx.names[level_number]
    data_dict = {}
    for k in variables:
        data_dict[k] = self.sel({variable_dim: k}, drop=True).squeeze(drop=True)
    return Dataset(data_dict)