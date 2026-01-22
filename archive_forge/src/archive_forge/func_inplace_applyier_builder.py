import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def inplace_applyier_builder(cls, key, func=None):
    """
        Bind actual aggregation function to the GroupBy aggregation method.

        Parameters
        ----------
        key : callable
            Function that takes GroupBy object and evaluates passed aggregation function.
        func : callable or str, optional
            Function that takes DataFrame and aggregate its data. Will be applied
            to each group at the grouped frame.

        Returns
        -------
        callable,
            Function that executes aggregation under GroupBy object.
        """
    inplace_args = [] if func is None else [func]

    def inplace_applyier(grp, *func_args, **func_kwargs):
        return key(grp, *inplace_args, *func_args, **func_kwargs)
    return inplace_applyier