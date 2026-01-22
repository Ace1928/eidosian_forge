import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.errors import MergeError
from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.pandas.metadata import ModinDtypes
from .utils import merge_partitioning
@classmethod
def range_partitioning_merge(cls, left, right, kwargs):
    """
        Execute merge using range-partitioning implementation.

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
    if kwargs.get('left_index', False) or kwargs.get('right_index', False) or kwargs.get('left_on', None) is not None or (kwargs.get('left_on', None) is not None) or (kwargs.get('how', 'left') not in ('left', 'inner')):
        raise NotImplementedError(f'The passed parameters are not yet supported by range-partitioning merge: kwargs={kwargs!r}')
    on = kwargs.get('on', None)
    if on is not None and (not isinstance(on, list)):
        on = [on]
    if on is None or len(on) > 1:
        raise NotImplementedError(f'Merging on multiple columns is not yet supported by range-partitioning merge: on={on!r}')
    if any((col not in left.columns or col not in right.columns for col in on)):
        raise NotImplementedError('Merging on an index level is not yet supported by range-partitioning merge.')

    def func(left, right):
        return left.merge(right, **kwargs)
    new_columns, new_dtypes = cls._compute_result_metadata(left, right, on, left_on=None, right_on=None, suffixes=kwargs.get('suffixes', ('_x', '_y')))
    return left.__constructor__(left._modin_frame._apply_func_to_range_partitioning_broadcast(right._modin_frame, func=func, key=on, new_columns=new_columns, new_dtypes=new_dtypes)).reset_index(drop=True)