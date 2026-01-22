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
def drop_vars(self, names: str | Iterable[Hashable] | Callable[[Self], str | Iterable[Hashable]], *, errors: ErrorOptions='raise') -> Self:
    """Drop variables from this dataset.

        Parameters
        ----------
        names : Hashable or iterable of Hashable or Callable
            Name(s) of variables to drop. If a Callable, this object is passed as its
            only argument and its result is used.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            dataset are dropped and no error is raised.

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[25.5, 26.3], [27.1, 28.0]]],
        ...         ),
        ...         "humidity": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[65.0, 63.8], [58.2, 59.6]]],
        ...         ),
        ...         "wind_speed": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[10.2, 8.5], [12.1, 9.8]]],
        ...         ),
        ...     },
        ...     coords={
        ...         "time": pd.date_range("2023-07-01", periods=1),
        ...         "latitude": [40.0, 40.2],
        ...         "longitude": [-75.0, -74.8],
        ...     },
        ... )
        >>> dataset
        <xarray.Dataset> Size: 136B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop the 'humidity' variable

        >>> dataset.drop_vars(["humidity"])
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop the 'humidity', 'temperature' variables

        >>> dataset.drop_vars(["humidity", "temperature"])
        <xarray.Dataset> Size: 72B
        Dimensions:     (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time        (time) datetime64[ns] 8B 2023-07-01
          * latitude    (latitude) float64 16B 40.0 40.2
          * longitude   (longitude) float64 16B -75.0 -74.8
        Data variables:
            wind_speed  (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop all indexes

        >>> dataset.drop_vars(lambda x: x.indexes)
        <xarray.Dataset> Size: 96B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Dimensions without coordinates: time, latitude, longitude
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Attempt to drop non-existent variable with errors="ignore"

        >>> dataset.drop_vars(["pressure"], errors="ignore")
        <xarray.Dataset> Size: 136B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Attempt to drop non-existent variable with errors="raise"

        >>> dataset.drop_vars(["pressure"], errors="raise")
        Traceback (most recent call last):
        ValueError: These variables cannot be found in this dataset: ['pressure']

        Raises
        ------
        ValueError
             Raised if you attempt to drop a variable which is not present, and the kwarg ``errors='raise'``.

        Returns
        -------
        dropped : Dataset

        See Also
        --------
        DataArray.drop_vars

        """
    if callable(names):
        names = names(self)
    if is_scalar(names) or not isinstance(names, Iterable):
        names_set = {names}
    else:
        names_set = set(names)
    if errors == 'raise':
        self._assert_all_in_dataset(names_set)
    other_names = set()
    for var in names_set:
        maybe_midx = self._indexes.get(var, None)
        if isinstance(maybe_midx, PandasMultiIndex):
            idx_coord_names = set(list(maybe_midx.index.names) + [maybe_midx.dim])
            idx_other_names = idx_coord_names - set(names_set)
            other_names.update(idx_other_names)
    if other_names:
        names_set |= set(other_names)
        warnings.warn(f'Deleting a single level of a MultiIndex is deprecated. Previously, this deleted all levels of a MultiIndex. Please also drop the following variables: {other_names!r} to avoid an error in the future.', DeprecationWarning, stacklevel=2)
    assert_no_index_corrupted(self.xindexes, names_set)
    variables = {k: v for k, v in self._variables.items() if k not in names_set}
    coord_names = {k for k in self._coord_names if k in variables}
    indexes = {k: v for k, v in self._indexes.items() if k not in names_set}
    return self._replace_with_new_dims(variables, coord_names=coord_names, indexes=indexes)