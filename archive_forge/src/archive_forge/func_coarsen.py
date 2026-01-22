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
def coarsen(self, dim: Mapping[Any, int] | None=None, boundary: CoarsenBoundaryOptions='exact', side: SideOptions | Mapping[Any, SideOptions]='left', coord_func: str | Callable | Mapping[Any, str | Callable]='mean', **window_kwargs: int) -> DatasetCoarsen:
    """
        Coarsen object for Datasets.

        Parameters
        ----------
        dim : mapping of hashable to int, optional
            Mapping from the dimension name to the window size.
        boundary : {"exact", "trim", "pad"}, default: "exact"
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of the window size. If 'trim', the excess entries are
            dropped. If 'pad', NA will be padded.
        side : {"left", "right"} or mapping of str to {"left", "right"}, default: "left"
        coord_func : str or mapping of hashable to str, default: "mean"
            function (name) that is applied to the coordinates,
            or a mapping from coordinate name to function (name).

        Returns
        -------
        core.rolling.DatasetCoarsen

        See Also
        --------
        core.rolling.DatasetCoarsen
        DataArray.coarsen

        :ref:`reshape.coarsen`
            User guide describing :py:func:`~xarray.Dataset.coarsen`

        :ref:`compute.coarsen`
            User guide on block arrgragation :py:func:`~xarray.Dataset.coarsen`

        :doc:`xarray-tutorial:fundamentals/03.3_windowed`
            Tutorial on windowed computation using :py:func:`~xarray.Dataset.coarsen`

        """
    from xarray.core.rolling import DatasetCoarsen
    dim = either_dict_or_kwargs(dim, window_kwargs, 'coarsen')
    return DatasetCoarsen(self, dim, boundary=boundary, side=side, coord_func=coord_func)