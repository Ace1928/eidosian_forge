from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import (
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args
def to_dask_dataframe(self, dim_order: Sequence[Hashable] | None=None, set_index: bool=False) -> DaskDataFrame:
    """
        Convert this dataset into a dask.dataframe.DataFrame.

        The dimensions, coordinates and data variables in this dataset form
        the columns of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting dask
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, default: False
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames do not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """
    import dask.array as da
    import dask.dataframe as dd
    ordered_dims = self._normalize_dim_order(dim_order=dim_order)
    columns = list(ordered_dims)
    columns.extend((k for k in self.coords if k not in self.dims))
    columns.extend(self.data_vars)
    ds_chunks = self.chunks
    series_list = []
    df_meta = pd.DataFrame()
    for name in columns:
        try:
            var = self.variables[name]
        except KeyError:
            size = self.sizes[name]
            data = da.arange(size, chunks=size, dtype=np.int64)
            var = Variable((name,), data)
        if isinstance(var, IndexVariable):
            var = var.to_base_variable()
        if not is_duck_dask_array(var._data):
            var = var.chunk()
        var_new_dims = var.set_dims(ordered_dims).chunk(ds_chunks)
        dask_array = var_new_dims._data.reshape(-1)
        series = dd.from_dask_array(dask_array, columns=name, meta=df_meta)
        series_list.append(series)
    df = dd.concat(series_list, axis=1)
    if set_index:
        dim_order = [*ordered_dims]
        if len(dim_order) == 1:
            dim, = dim_order
            df = df.set_index(dim)
        else:
            df = df.set_index(dim_order)
    return df