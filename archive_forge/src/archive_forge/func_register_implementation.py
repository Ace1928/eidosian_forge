import pandas
from modin.core.dataframe.pandas.metadata import ModinIndex
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default2pandas.groupby import GroupBy, GroupByDefault
from .tree_reduce import TreeReduce
@classmethod
def register_implementation(cls, map_func, reduce_func):
    """
        Register callables to be recognized as an implementations of tree-reduce phases.

        Parameters
        ----------
        map_func : callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Callable to register.
        reduce_func : callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Callable to register.
        """
    setattr(map_func, cls._GROUPBY_REDUCE_IMPL_FLAG, True)
    setattr(reduce_func, cls._GROUPBY_REDUCE_IMPL_FLAG, True)