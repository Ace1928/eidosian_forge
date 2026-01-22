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
def interp_like(self, other: T_Xarray, method: InterpOptions='linear', assume_sorted: bool=False, kwargs: Mapping[str, Any] | None=None, method_non_numeric: str='nearest') -> Self:
    """Interpolate this object onto the coordinates of another object,
        filling the out of range values with NaN.

        If interpolating along a single existing dimension,
        :py:class:`scipy.interpolate.interp1d` is called. When interpolating
        along multiple existing dimensions, an attempt is made to decompose the
        interpolation into multiple 1-dimensional interpolations. If this is
        possible, :py:class:`scipy.interpolate.interp1d` is called. Otherwise,
        :py:func:`scipy.interpolate.interpn` is called.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset. Missing values are skipped.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation. Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krogh', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        assume_sorted : bool, default: False
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword passed to scipy's interpolator.
        method_non_numeric : {"nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method for non-numeric types. Passed on to :py:meth:`Dataset.reindex`.
            ``"nearest"`` is used by default.

        Returns
        -------
        interpolated : Dataset
            Another dataset by interpolating this dataset's data along the
            coordinates of the other object.

        Notes
        -----
        scipy is required.
        If the dataset has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        Dataset.interp
        Dataset.reindex_like
        """
    if kwargs is None:
        kwargs = {}
    coords = {}
    other_indexes = other.xindexes
    for dim in self.dims:
        other_dim_coords = other_indexes.get_all_coords(dim, errors='ignore')
        if len(other_dim_coords) == 1:
            coords[dim] = other_dim_coords[dim]
    numeric_coords: dict[Hashable, pd.Index] = {}
    object_coords: dict[Hashable, pd.Index] = {}
    for k, v in coords.items():
        if v.dtype.kind in 'uifcMm':
            numeric_coords[k] = v
        else:
            object_coords[k] = v
    ds = self
    if object_coords:
        ds = self.reindex(object_coords)
    return ds.interp(coords=numeric_coords, method=method, assume_sorted=assume_sorted, kwargs=kwargs, method_non_numeric=method_non_numeric)