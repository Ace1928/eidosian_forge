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
def set_xindex(self, coord_names: str | Sequence[Hashable], index_cls: type[Index] | None=None, **options) -> Self:
    """Set a new, Xarray-compatible index from one or more existing
        coordinate(s).

        Parameters
        ----------
        coord_names : str or list
            Name(s) of the coordinate(s) used to build the index.
            If several names are given, their order matters.
        index_cls : subclass of :class:`~xarray.indexes.Index`, optional
            The type of index to create. By default, try setting
            a ``PandasIndex`` if ``len(coord_names) == 1``,
            otherwise a ``PandasMultiIndex``.
        **options
            Options passed to the index constructor.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data and with a new index.

        """
    if is_scalar(coord_names) or not isinstance(coord_names, Sequence):
        coord_names = [coord_names]
    if index_cls is None:
        if len(coord_names) == 1:
            index_cls = PandasIndex
        else:
            index_cls = PandasMultiIndex
    elif not issubclass(index_cls, Index):
        raise TypeError(f'{index_cls} is not a subclass of xarray.Index')
    invalid_coords = set(coord_names) - self._coord_names
    if invalid_coords:
        msg = ['invalid coordinate(s)']
        no_vars = invalid_coords - set(self._variables)
        data_vars = invalid_coords - no_vars
        if no_vars:
            msg.append(f"those variables don't exist: {no_vars}")
        if data_vars:
            msg.append(f'those variables are data variables: {data_vars}, use `set_coords` first')
        raise ValueError('\n'.join(msg))
    indexed_coords = set(coord_names) & set(self._indexes)
    if indexed_coords:
        raise ValueError(f'those coordinates already have an index: {indexed_coords}')
    coord_vars = {name: self._variables[name] for name in coord_names}
    index = index_cls.from_variables(coord_vars, options=options)
    new_coord_vars = index.create_variables(coord_vars)
    if isinstance(index, PandasMultiIndex):
        coord_names = [index.dim] + list(coord_names)
    variables: dict[Hashable, Variable]
    indexes: dict[Hashable, Index]
    if len(coord_names) == 1:
        variables = self._variables.copy()
        indexes = self._indexes.copy()
        name = list(coord_names).pop()
        if name in new_coord_vars:
            variables[name] = new_coord_vars[name]
        indexes[name] = index
    else:
        variables = {}
        for name, var in self._variables.items():
            if name not in coord_names:
                variables[name] = var
        indexes = {}
        for name, idx in self._indexes.items():
            if name not in coord_names:
                indexes[name] = idx
        for name in coord_names:
            try:
                variables[name] = new_coord_vars[name]
            except KeyError:
                variables[name] = self._variables[name]
            indexes[name] = index
    return self._replace(variables=variables, coord_names=self._coord_names | set(coord_names), indexes=indexes)