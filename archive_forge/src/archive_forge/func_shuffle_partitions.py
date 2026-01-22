import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
import pandas
from pandas._libs.lib import no_default
from modin.config import (
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
@classmethod
@wait_computations_if_benchmark_mode
def shuffle_partitions(cls, partitions, index, shuffle_functions: 'ShuffleFunctions', final_shuffle_func, right_partitions=None):
    """
        Return shuffled partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to shuffle.
        index : int or list of ints
            The index(es) of the column partitions corresponding to the partitions that contain the column to sample.
        shuffle_functions : ShuffleFunctions
            An object implementing the functions that we will be using to perform this shuffle.
        final_shuffle_func : Callable(pandas.DataFrame) -> pandas.DataFrame
            Function that shuffles the data within each new partition.
        right_partitions : np.ndarray, optional
            Partitions to broadcast to `self` partitions. If specified, the method builds range-partitioning
            for `right_partitions` basing on bins calculated for `partitions`, then performs broadcasting.

        Returns
        -------
        np.ndarray
            A list of row-partitions that have been shuffled.
        """
    masked_partitions = partitions[:, index]
    sample_func = cls.preprocess_func(shuffle_functions.sample_fn)
    if masked_partitions.ndim == 1:
        samples = [partition.apply(sample_func) for partition in masked_partitions]
    else:
        samples = [cls._row_partition_class(row_part, full_axis=False).apply(sample_func) for row_part in masked_partitions]
    samples = cls.get_objects_from_partitions(samples)
    num_bins = shuffle_functions.pivot_fn(samples)
    row_partitions = cls.row_partitions(partitions)
    if num_bins > 1:
        split_row_partitions = np.array([partition.split(shuffle_functions.split_fn, num_splits=num_bins, extract_metadata=False) for partition in row_partitions]).T
        if right_partitions is None:
            return np.array([[cls._column_partitions_class(row_partition, full_axis=False).apply(final_shuffle_func)] for row_partition in split_row_partitions])
        right_row_parts = cls.row_partitions(right_partitions)
        right_split_row_partitions = np.array([partition.split(shuffle_functions.split_fn, num_splits=num_bins, extract_metadata=False) for partition in right_row_parts]).T
        return np.array([cls._column_partitions_class(row_partition, full_axis=False).apply(final_shuffle_func, other_axis_partition=cls._column_partitions_class(right_row_partitions)) for right_row_partitions, row_partition in zip(right_split_row_partitions, split_row_partitions)])
    else:
        if right_partitions is None:
            return np.array([row_part.apply(final_shuffle_func) for row_part in row_partitions])
        right_row_parts = cls.row_partitions(right_partitions)
        return np.array([row_part.apply(final_shuffle_func, other_axis_partition=right_row_part) for right_row_part, row_part in zip(right_row_parts, row_partitions)])