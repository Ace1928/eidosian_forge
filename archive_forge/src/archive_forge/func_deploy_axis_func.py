import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
@classmethod
def deploy_axis_func(cls, axis, func, f_args, f_kwargs, num_splits, maintain_partitioning, *partitions, min_block_size, lengths=None, manual_partition=False, return_generator=False):
    """
        Deploy a function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        min_block_size : int
            Minimum number of rows/columns in a single split.
        lengths : list, optional
            The list of lengths to shuffle the object.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
        return_generator : bool, default: False
            Return a generator from the function, set to `True` for Ray backend
            as Ray remote functions can return Generators.

        Returns
        -------
        list | Generator
            A list or generator of pandas DataFrames.
        """
    len_partitions = len(partitions)
    lengths_partitions = [len(part) for part in partitions]
    widths_partitions = [len(part.columns) for part in partitions]
    dataframe = pandas.concat(list(partitions), axis=axis, copy=False)
    del partitions
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        try:
            result = func(dataframe, *f_args, **f_kwargs)
        except ValueError as err:
            if 'assignment destination is read-only' in str(err):
                result = func(dataframe.copy(), *f_args, **f_kwargs)
            else:
                raise err
    del dataframe
    if num_splits == 1:
        lengths = None
    elif manual_partition:
        lengths = list(lengths)
    elif num_splits != len_partitions or not maintain_partitioning:
        lengths = None
    elif axis == 0:
        lengths = lengths_partitions
        if sum(lengths) != len(result):
            lengths = None
    else:
        lengths = widths_partitions
        if sum(lengths) != len(result.columns):
            lengths = None
    if return_generator:
        return generate_result_of_axis_func_pandas(axis, num_splits, result, min_block_size, lengths)
    else:
        return split_result_of_axis_func_pandas(axis, num_splits, result, min_block_size, lengths)