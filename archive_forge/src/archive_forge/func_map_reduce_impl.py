import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@classmethod
def map_reduce_impl(cls, qc, unique_keys, drop_column_level, pivot_kwargs):
    """Compute 'pivot_table()' using MapReduce implementation."""
    if pivot_kwargs['margins']:
        raise NotImplementedError("MapReduce 'pivot_table' implementation doesn't support 'margins=True' parameter")
    index, columns, values = (pivot_kwargs['index'], pivot_kwargs['columns'], pivot_kwargs['values'])
    aggfunc = pivot_kwargs['aggfunc']
    if not GroupbyReduceImpl.has_impl_for(aggfunc):
        raise NotImplementedError("MapReduce 'pivot_table' implementation only supports 'aggfuncs' that are implemented in 'GroupbyReduceImpl'")
    if len(set(index).intersection(columns)) > 0:
        raise NotImplementedError("MapReduce 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'")
    to_group, keys_columns = cls._separate_data_from_grouper(qc, values, unique_keys)
    to_unstack = columns if index else None
    result = GroupbyReduceImpl.build_qc_method(aggfunc, finalizer_fn=lambda df: cls._pivot_table_from_groupby(df, pivot_kwargs['dropna'], drop_column_level, to_unstack, pivot_kwargs['fill_value']))(to_group, by=keys_columns, axis=0, groupby_kwargs={'observed': pivot_kwargs['observed'], 'sort': pivot_kwargs['sort']}, agg_args=(), agg_kwargs={}, drop=True)
    if to_unstack is None:
        result = result.transpose()
    return result