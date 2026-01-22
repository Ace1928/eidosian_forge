from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def process_axes(self, obj, selection: Selection, columns=None) -> DataFrame:
    """process axes filters"""
    if columns is not None:
        columns = list(columns)
    if columns is not None and self.is_multi_index:
        assert isinstance(self.levels, list)
        for n in self.levels:
            if n not in columns:
                columns.insert(0, n)
    for axis, labels in self.non_index_axes:
        obj = _reindex_axis(obj, axis, labels, columns)

        def process_filter(field, filt, op):
            for axis_name in obj._AXIS_ORDERS:
                axis_number = obj._get_axis_number(axis_name)
                axis_values = obj._get_axis(axis_name)
                assert axis_number is not None
                if field == axis_name:
                    if self.is_multi_index:
                        filt = filt.union(Index(self.levels))
                    takers = op(axis_values, filt)
                    return obj.loc(axis=axis_number)[takers]
                elif field in axis_values:
                    values = ensure_index(getattr(obj, field).values)
                    filt = ensure_index(filt)
                    if isinstance(obj, DataFrame):
                        axis_number = 1 - axis_number
                    takers = op(values, filt)
                    return obj.loc(axis=axis_number)[takers]
            raise ValueError(f'cannot find the field [{field}] for filtering!')
    if selection.filter is not None:
        for field, op, filt in selection.filter.format():
            obj = process_filter(field, filt, op)
    return obj