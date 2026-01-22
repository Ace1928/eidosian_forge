import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@classmethod
def range_partition_impl(cls, qc, unique_keys, drop_column_level, pivot_kwargs):
    """Compute 'pivot_table()' using Range-Partitioning implementation."""
    if pivot_kwargs['margins']:
        raise NotImplementedError("Range-partitioning 'pivot_table' implementation doesn't support 'margins=True' parameter")
    index, columns, values = (pivot_kwargs['index'], pivot_kwargs['columns'], pivot_kwargs['values'])
    if len(set(index).intersection(columns)) > 0:
        raise NotImplementedError("Range-partitioning 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'")
    if values is not None:
        to_take = list(np.unique(list(index) + list(columns) + list(values)))
        qc = qc.getitem_column_array(to_take, ignore_order=True)
    to_unstack = columns if index else None
    groupby_result = qc._groupby_shuffle(by=list(unique_keys), agg_func=pivot_kwargs['aggfunc'], axis=0, groupby_kwargs={'observed': pivot_kwargs['observed'], 'sort': pivot_kwargs['sort']}, agg_args=(), agg_kwargs={}, drop=True)
    result = groupby_result._modin_frame.apply_full_axis(axis=0, func=lambda df: cls._pivot_table_from_groupby(df, pivot_kwargs['dropna'], drop_column_level, to_unstack, pivot_kwargs['fill_value'], sort=pivot_kwargs['sort'] if len(unique_keys) > 1 else False))
    if to_unstack is None:
        result = result.transpose()
    return qc.__constructor__(result)