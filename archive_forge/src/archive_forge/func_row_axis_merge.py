import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.errors import MergeError
from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.pandas.metadata import ModinDtypes
from .utils import merge_partitioning
@classmethod
def row_axis_merge(cls, left, right, kwargs):
    """
        Execute merge using row-axis implementation.

        Parameters
        ----------
        left : PandasQueryCompiler
        right : PandasQueryCompiler
        kwargs : dict
            Keyword arguments for ``pandas.merge()`` function.

        Returns
        -------
        PandasQueryCompiler
        """
    how = kwargs.get('how', 'inner')
    on = kwargs.get('on', None)
    left_on = kwargs.get('left_on', None)
    right_on = kwargs.get('right_on', None)
    left_index = kwargs.get('left_index', False)
    right_index = kwargs.get('right_index', False)
    sort = kwargs.get('sort', False)
    right_to_broadcast = right._modin_frame.combine()
    if how in ['left', 'inner'] and left_index is False and (right_index is False):
        kwargs['sort'] = False

        def should_keep_index(left, right):
            keep_index = False
            if left_on is not None and right_on is not None:
                keep_index = any((o in left.index.names and o in right_on and (o in right.index.names) for o in left_on))
            elif on is not None:
                keep_index = any((o in left.index.names and o in right.index.names for o in on))
            return keep_index

        def map_func(left, right, *axis_lengths, kwargs=kwargs, **service_kwargs):
            df = pandas.merge(left, right, **kwargs)
            if kwargs['how'] == 'left':
                partition_idx = service_kwargs['partition_idx']
                if len(axis_lengths):
                    if not should_keep_index(left, right):
                        start = sum(axis_lengths[:partition_idx])
                        stop = sum(axis_lengths[:partition_idx + 1])
                        df.index = pandas.RangeIndex(start, stop)
            return df
        if left_on is not None and right_on is not None:
            left_on = list(left_on) if is_list_like(left_on) else [left_on]
            right_on = list(right_on) if is_list_like(right_on) else [right_on]
        elif on is not None:
            on = list(on) if is_list_like(on) else [on]
        new_columns, new_dtypes = cls._compute_result_metadata(left, right, on, left_on, right_on, kwargs.get('suffixes', ('_x', '_y')))
        new_left = left.__constructor__(left._modin_frame.broadcast_apply_full_axis(axis=1, func=map_func, enumerate_partitions=how == 'left', other=right_to_broadcast, keep_partitioning=False, num_splits=merge_partitioning(left._modin_frame, right._modin_frame, axis=1), new_columns=new_columns, sync_labels=False, dtypes=new_dtypes, pass_axis_lengths_to_partitions=how == 'left'))
        keep_index = False
        if left._modin_frame.has_materialized_index:
            keep_index = should_keep_index(left, right)
        elif left_on is not None and right_on is not None:
            keep_index = any((o not in right.columns and o in left_on and (o not in left.columns) for o in right_on))
        elif on is not None:
            keep_index = any((o not in right.columns and o not in left.columns for o in on))
        if sort:
            if left_on is not None and right_on is not None:
                new_left = new_left.sort_index(axis=0, level=left_on + right_on) if keep_index else new_left.sort_rows_by_column_values(left_on + right_on)
            elif on is not None:
                new_left = new_left.sort_index(axis=0, level=on) if keep_index else new_left.sort_rows_by_column_values(on)
        return new_left.reset_index(drop=True) if not keep_index and (kwargs['how'] != 'left' or sort) else new_left
    else:
        return left.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)