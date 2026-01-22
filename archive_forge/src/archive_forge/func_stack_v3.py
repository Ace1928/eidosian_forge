from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame:
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError('Columns with duplicate values are not supported in stack')
    drop_levnums = sorted(level, reverse=True)
    stack_cols = frame.columns._drop_level_numbers([k for k in range(frame.columns.nlevels) if k not in level][::-1])
    if len(level) > 1:
        sorter = np.argsort(level)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols
    stack_cols_unique = stack_cols.unique()
    ordered_stack_cols_unique = ordered_stack_cols.unique()
    buf = []
    for idx in stack_cols_unique:
        if len(frame.columns) == 1:
            data = frame.copy()
        else:
            if len(level) == 1:
                idx = (idx,)
            gen = iter(idx)
            column_indexer = tuple((next(gen) if k in level else slice(None) for k in range(frame.columns.nlevels)))
            data = frame.loc[:, column_indexer]
        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = RangeIndex(len(data.columns))
        buf.append(data)
    result: Series | DataFrame
    if len(buf) > 0 and (not frame.empty):
        result = concat(buf)
        ratio = len(result) // len(frame)
    else:
        if len(level) < frame.columns.nlevels:
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)
        ratio = 0
    if len(level) < frame.columns.nlevels:
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]
    index_levels: list | FrozenList
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))
    if isinstance(stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols.unique()]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]
    result.index = MultiIndex(levels=index_levels + column_levels, codes=index_codes + column_codes, names=frame.index.names + list(ordered_stack_cols.names), verify_integrity=False)
    len_df = len(frame)
    n_uniques = len(ordered_stack_cols_unique)
    indexer = np.arange(n_uniques)
    idxs = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)
    result = result.take(idxs)
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:
            result = Series(index=result.index)
        else:
            result = result.iloc[:, 0]
    if result.ndim == 1:
        result.name = None
    return result