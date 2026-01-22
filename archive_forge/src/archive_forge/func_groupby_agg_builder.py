import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def groupby_agg_builder(df, by=None, drop=False, partition_idx=None):
    """
            Compute groupby aggregation for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            by : pandas.DataFrame, optional
                Broadcasted partition which contains `by` columns.
            drop : bool, default: False
                Indicates whether `by` partition came from the `self` frame.
            partition_idx : int, optional
                Positional partition index along groupby axis.

            Returns
            -------
            pandas.DataFrame
                DataFrame containing the result of groupby aggregation
                for this particular partition.
            """
    groupby_kwargs['as_index'] = True
    partition_agg_func = GroupByReduce.get_callable(agg_func, df)
    internal_by_cols = pandas.Index([])
    missed_by_cols = pandas.Index([])
    if by is not None:
        internal_by_df = by[internal_by]
        if isinstance(internal_by_df, pandas.Series):
            internal_by_df = internal_by_df.to_frame()
        missed_by_cols = internal_by_df.columns.difference(df.columns)
        if len(missed_by_cols) > 0:
            df = pandas.concat([df, internal_by_df[missed_by_cols]], axis=1, copy=False)
        internal_by_cols = internal_by_df.columns
        external_by = by.columns.difference(internal_by).unique()
        external_by_df = by[external_by].squeeze(axis=1)
        if isinstance(external_by_df, pandas.DataFrame):
            external_by_cols = [o for _, o in external_by_df.items()]
        else:
            external_by_cols = [external_by_df]
        by = internal_by_cols.tolist() + external_by_cols
    else:
        by = []
    by += not_broadcastable_by
    level = groupby_kwargs.get('level', None)
    if level is not None and (not by):
        by = None
        by_length = len(level) if is_list_like(level) else 1
    else:
        by_length = len(by)

    def compute_groupby(df, drop=False, partition_idx=0):
        """Compute groupby aggregation for a single partition."""
        target_df = df.squeeze(axis=1) if series_groupby else df
        grouped_df = target_df.groupby(by=by, axis=axis, **groupby_kwargs)
        try:
            result = partition_agg_func(grouped_df, *agg_args, **agg_kwargs)
        except DataError:
            result = pandas.DataFrame(index=grouped_df.size().index)
        if isinstance(result, pandas.Series):
            result = result.to_frame(result.name if result.name is not None else MODIN_UNNAMED_SERIES_LABEL)
        selection = agg_func.keys() if isinstance(agg_func, dict) else None
        if selection is None:
            misaggregated_cols = missed_by_cols.intersection(result.columns)
        else:
            misaggregated_cols = []
        if not as_index:
            GroupBy.handle_as_index_for_dataframe(result, internal_by_cols, by_cols_dtypes=df[internal_by_cols].dtypes.values, by_length=by_length, selection=selection, partition_idx=partition_idx, drop=drop, inplace=True, method='transform' if is_transform_method else None)
        else:
            new_index_names = tuple((None if isinstance(name, str) and name.startswith(MODIN_UNNAMED_SERIES_LABEL) else name for name in result.index.names))
            result.index.names = new_index_names
        if len(misaggregated_cols) > 0:
            result.drop(columns=misaggregated_cols, inplace=True)
        return result
    try:
        return compute_groupby(df, drop, partition_idx)
    except (ValueError, KeyError):
        return compute_groupby(df.copy(), drop, partition_idx)