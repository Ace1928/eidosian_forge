import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
@classmethod
def deploy_splitting_func(cls, axis, split_func, f_args, f_kwargs, num_splits, *partitions, extract_metadata=False):
    """
        Deploy a splitting function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        split_func : callable(pandas.DataFrame) -> list[pandas.DataFrame]
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to `split_func`.
        f_kwargs : dict
            Keyword arguments to pass to `split_func`.
        num_splits : int
            The number of splits the `split_func` return.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        extract_metadata : bool, default: False
            Whether to return metadata (length, width, ip) of the result. Note that `True` value
            is not supported in `PandasDataframeAxisPartition` class.

        Returns
        -------
        list
            A list of pandas DataFrames.
        """
    dataframe = pandas.concat(list(partitions), axis=axis, copy=False)
    del partitions
    return split_func(dataframe, *f_args, **f_kwargs)