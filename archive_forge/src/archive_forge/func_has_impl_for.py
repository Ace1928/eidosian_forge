import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@classmethod
def has_impl_for(cls, agg_func):
    """
        Check whether the class has TreeReduce implementation for the specified `agg_func`.

        Parameters
        ----------
        agg_func : hashable or dict

        Returns
        -------
        bool
        """
    if hashable(agg_func):
        return agg_func in cls._groupby_reduce_impls
    if not isinstance(agg_func, dict):
        return False
    from modin.pandas.utils import walk_aggregation_dict
    for _, func, _, _ in walk_aggregation_dict(agg_func):
        if func not in cls._groupby_reduce_impls:
            return False
    return True