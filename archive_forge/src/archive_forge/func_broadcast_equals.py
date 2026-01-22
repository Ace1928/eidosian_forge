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
def broadcast_equals(self, other: Self) -> bool:
    """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        Examples
        --------

        # 2D array with shape (1, 3)

        >>> data = np.array([[1, 2, 3]])
        >>> a = xr.Dataset(
        ...     {"variable_name": (("space", "time"), data)},
        ...     coords={"space": [0], "time": [0, 1, 2]},
        ... )
        >>> a
        <xarray.Dataset> Size: 56B
        Dimensions:        (space: 1, time: 3)
        Coordinates:
          * space          (space) int64 8B 0
          * time           (time) int64 24B 0 1 2
        Data variables:
            variable_name  (space, time) int64 24B 1 2 3

        # 2D array with shape (3, 1)

        >>> data = np.array([[1], [2], [3]])
        >>> b = xr.Dataset(
        ...     {"variable_name": (("time", "space"), data)},
        ...     coords={"time": [0, 1, 2], "space": [0]},
        ... )
        >>> b
        <xarray.Dataset> Size: 56B
        Dimensions:        (time: 3, space: 1)
        Coordinates:
          * time           (time) int64 24B 0 1 2
          * space          (space) int64 8B 0
        Data variables:
            variable_name  (time, space) int64 24B 1 2 3

        .equals returns True if two Datasets have the same values, dimensions, and coordinates. .broadcast_equals returns True if the
        results of broadcasting two Datasets against each other have the same values, dimensions, and coordinates.

        >>> a.equals(b)
        False

        >>> a.broadcast_equals(b)
        True

        >>> a2, b2 = xr.broadcast(a, b)
        >>> a2.equals(b2)
        True

        See Also
        --------
        Dataset.equals
        Dataset.identical
        Dataset.broadcast
        """
    try:
        return self._all_compat(other, 'broadcast_equals')
    except (TypeError, AttributeError):
        return False