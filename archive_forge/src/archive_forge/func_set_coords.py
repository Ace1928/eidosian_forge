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
def set_coords(self, names: Hashable | Iterable[Hashable]) -> Self:
    """Given names of one or more variables, set them as coordinates

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables in this dataset to convert into coordinates.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "pressure": ("time", [1.013, 1.2, 3.5]),
        ...         "time": pd.date_range("2023-01-01", periods=3),
        ...     }
        ... )
        >>> dataset
        <xarray.Dataset> Size: 48B
        Dimensions:   (time: 3)
        Coordinates:
          * time      (time) datetime64[ns] 24B 2023-01-01 2023-01-02 2023-01-03
        Data variables:
            pressure  (time) float64 24B 1.013 1.2 3.5

        >>> dataset.set_coords("pressure")
        <xarray.Dataset> Size: 48B
        Dimensions:   (time: 3)
        Coordinates:
            pressure  (time) float64 24B 1.013 1.2 3.5
          * time      (time) datetime64[ns] 24B 2023-01-01 2023-01-02 2023-01-03
        Data variables:
            *empty*

        On calling ``set_coords`` , these data variables are converted to coordinates, as shown in the final dataset.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.swap_dims
        Dataset.assign_coords
        """
    if isinstance(names, str) or not isinstance(names, Iterable):
        names = [names]
    else:
        names = list(names)
    self._assert_all_in_dataset(names)
    obj = self.copy()
    obj._coord_names.update(names)
    return obj