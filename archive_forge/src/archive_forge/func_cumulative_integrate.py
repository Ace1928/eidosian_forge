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
def cumulative_integrate(self, coord: Hashable | Sequence[Hashable], datetime_unit: DatetimeUnitOptions=None) -> Self:
    """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

            The first entry of the cumulative integral of each variable is always 0, in
            order to keep the length of the dimension unchanged between input and
            output.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns',                         'ps', 'fs', 'as', None}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.cumulative_integrate
        scipy.integrate.cumulative_trapezoid : corresponding scipy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) int64 32B 5 5 6 6
            b        (x) int64 32B 1 2 1 0
        >>> ds.cumulative_integrate("x")
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) float64 32B 0.0 5.0 10.5 16.5
            b        (x) float64 32B 0.0 1.5 3.0 3.5
        >>> ds.cumulative_integrate("y")
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) float64 32B 0.0 30.0 8.0 20.0
            b        (x) float64 32B 0.0 9.0 3.0 4.0
        """
    if not isinstance(coord, (list, tuple)):
        coord = (coord,)
    result = self
    for c in coord:
        result = result._integrate_one(c, datetime_unit=datetime_unit, cumulative=True)
    return result