import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@lazy_metadata_decorator(apply_axis='both')
def broadcast_apply_full_axis(self, axis, func, other, new_index=None, new_columns=None, apply_indices=None, enumerate_partitions=False, dtypes=None, keep_partitioning=True, num_splits=None, sync_labels=True, pass_axis_lengths_to_partitions=False):
    """
        Broadcast partitions of `other` Modin DataFrame and apply a function along full axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply over (0 - rows, 1 - columns).
        func : callable
            Function to apply.
        other : PandasDataframe or list
            Modin DataFrame(s) to broadcast.
        new_index : list-like, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            Columns of the result. We may know this in
            advance, and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether pass partition index into applied `func` or not.
            Note that `func` must be able to obtain `partition_idx` kwarg.
        dtypes : list-like, default: None
            Data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.
        keep_partitioning : boolean, default: True
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        sync_labels : boolean, default: True
            Synchronize external indexes (`new_index`, `new_columns`) with internal indexes.
            This could be used when you're certain that the indices in partitions are equal to
            the provided hints in order to save time on syncing them.
        pass_axis_lengths_to_partitions : bool, default: False
            Whether pass partition lengths along `axis ^ 1` to the kernel `func`.
            Note that `func` must be able to obtain `df, *axis_lengths`.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
    if other is not None:
        if not isinstance(other, list):
            other = [other]
        other = [o._extract_partitions() for o in other] if len(other) else None
    if apply_indices is not None:
        numeric_indices = self.get_axis(axis ^ 1).get_indexer_for(apply_indices)
        apply_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices).keys()
    apply_func_args = None
    if pass_axis_lengths_to_partitions:
        if axis == 0:
            apply_func_args = self._column_widths_cache if self._column_widths_cache is not None else [part.width(materialize=False) for part in self._partitions[0]]
        else:
            apply_func_args = self._row_lengths_cache if self._row_lengths_cache is not None else [part.length(materialize=False) for part in self._partitions.T[0]]
    new_partitions = self._partition_mgr_cls.broadcast_axis_partitions(axis=axis, left=self._partitions, right=other, apply_func=self._build_treereduce_func(axis, func), apply_indices=apply_indices, enumerate_partitions=enumerate_partitions, keep_partitioning=keep_partitioning, num_splits=num_splits, apply_func_args=apply_func_args)
    kw = {'row_lengths': None, 'column_widths': None}
    if isinstance(dtypes, str) and dtypes == 'copy':
        kw['dtypes'] = self.copy_dtypes_cache()
    elif isinstance(dtypes, DtypesDescriptor):
        kw['dtypes'] = ModinDtypes(dtypes)
    elif dtypes is not None:
        if isinstance(dtypes, (pandas.Series, ModinDtypes)):
            kw['dtypes'] = dtypes.copy()
        elif new_columns is None:
            kw['dtypes'] = ModinDtypes(DtypesDescriptor(remaining_dtype=pandas.api.types.pandas_dtype(dtypes)))
        else:
            kw['dtypes'] = pandas.Series(dtypes, index=new_columns) if is_list_like(dtypes) else pandas.Series([pandas.api.types.pandas_dtype(dtypes)] * len(new_columns), index=new_columns)
    is_index_materialized = ModinIndex.is_materialized_index(new_index)
    is_columns_materialized = ModinIndex.is_materialized_index(new_columns)
    if axis == 0:
        if is_columns_materialized and len(new_partitions.shape) > 1 and (new_partitions.shape[1] == 1):
            kw['column_widths'] = [len(new_columns)]
    elif axis == 1:
        if is_index_materialized and new_partitions.shape[0] == 1:
            kw['row_lengths'] = [len(new_index)]
    if not keep_partitioning:
        if kw['row_lengths'] is None and is_index_materialized:
            if axis == 0:
                kw['row_lengths'] = get_length_list(axis_len=len(new_index), num_splits=new_partitions.shape[0], min_block_size=MinPartitionSize.get())
            elif axis == 1:
                if self._row_lengths_cache is not None and len(new_index) == sum(self._row_lengths_cache):
                    kw['row_lengths'] = self._row_lengths_cache
        if kw['column_widths'] is None and is_columns_materialized:
            if axis == 1:
                kw['column_widths'] = get_length_list(axis_len=len(new_columns), num_splits=new_partitions.shape[1], min_block_size=MinPartitionSize.get())
            elif axis == 0:
                if self._column_widths_cache is not None and len(new_columns) == sum(self._column_widths_cache):
                    kw['column_widths'] = self._column_widths_cache
    elif axis == 0:
        if kw['row_lengths'] is None and self._row_lengths_cache is not None and is_index_materialized and (len(new_index) == sum(self._row_lengths_cache)) and all((r != 0 for r in self._row_lengths_cache)):
            kw['row_lengths'] = self._row_lengths_cache
    elif axis == 1:
        if kw['column_widths'] is None and self._column_widths_cache is not None and is_columns_materialized and (len(new_columns) == sum(self._column_widths_cache)) and all((w != 0 for w in self._column_widths_cache)):
            kw['column_widths'] = self._column_widths_cache
    result = self.__constructor__(new_partitions, index=new_index, columns=new_columns, **kw)
    if sync_labels and new_index is not None:
        result.synchronize_labels(axis=0)
    if sync_labels and new_columns is not None:
        result.synchronize_labels(axis=1)
    return result