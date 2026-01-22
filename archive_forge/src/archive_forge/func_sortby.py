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
def sortby(self, variables: Hashable | DataArray | Sequence[Hashable | DataArray] | Callable[[Self], Hashable | DataArray | list[Hashable | DataArray]], ascending: bool=True) -> Self:
    """
        Sort object by labels or values (along an axis).

        Sorts the dataset, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables : Hashable, DataArray, sequence of Hashable or DataArray, or Callable
            1D DataArray objects or name(s) of 1D variable(s) in coords whose values are
            used to sort this array. If a callable, the callable is passed this object,
            and the result is used as the value for cond.
        ascending : bool, default: True
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted : Dataset
            A new dataset where all the specified dims are sorted by dim
            labels.

        See Also
        --------
        DataArray.sortby
        numpy.sort
        pandas.sort_values
        pandas.sort_index

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": (("x", "y"), [[1, 2], [3, 4]]),
        ...         "B": (("x", "y"), [[5, 6], [7, 8]]),
        ...     },
        ...     coords={"x": ["b", "a"], "y": [1, 0]},
        ... )
        >>> ds.sortby("x")
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
          * y        (y) int64 16B 1 0
        Data variables:
            A        (x, y) int64 32B 3 4 1 2
            B        (x, y) int64 32B 7 8 5 6
        >>> ds.sortby(lambda x: -x["y"])
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * x        (x) <U1 8B 'b' 'a'
          * y        (y) int64 16B 1 0
        Data variables:
            A        (x, y) int64 32B 1 2 3 4
            B        (x, y) int64 32B 5 6 7 8
        """
    from xarray.core.dataarray import DataArray
    if callable(variables):
        variables = variables(self)
    if not isinstance(variables, list):
        variables = [variables]
    else:
        variables = variables
    arrays = [v if isinstance(v, DataArray) else self[v] for v in variables]
    aligned_vars = align(self, *arrays, join='left')
    aligned_self = cast('Self', aligned_vars[0])
    aligned_other_vars = cast(tuple[DataArray, ...], aligned_vars[1:])
    vars_by_dim = defaultdict(list)
    for data_array in aligned_other_vars:
        if data_array.ndim != 1:
            raise ValueError('Input DataArray is not 1-D.')
        key, = data_array.dims
        vars_by_dim[key].append(data_array)
    indices = {}
    for key, arrays in vars_by_dim.items():
        order = np.lexsort(tuple(reversed(arrays)))
        indices[key] = order if ascending else order[::-1]
    return aligned_self.isel(indices)