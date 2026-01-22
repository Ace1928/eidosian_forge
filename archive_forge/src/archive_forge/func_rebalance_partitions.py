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
def rebalance_partitions(cls, partitions):
    """
        Rebalance a 2-d array of partitions if we are using ``PandasOnRay`` or ``PandasOnDask`` executions.

        For all other executions, the partitions are returned unchanged.

        Rebalance the partitions by building a new array
        of partitions out of the original ones so that:

        - If all partitions have a length, each new partition has roughly the same number of rows.
        - Otherwise, each new partition spans roughly the same number of old partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to rebalance.

        Returns
        -------
        np.ndarray
            A NumPy array with the same; or new, rebalanced, partitions, depending on the execution
            engine and storage format.
        list[int] or None
            Row lengths if possible to compute it.
        """
    max_excess_of_num_partitions = 1.5
    num_existing_partitions = partitions.shape[0]
    ideal_num_new_partitions = NPartitions.get()
    if num_existing_partitions <= ideal_num_new_partitions * max_excess_of_num_partitions:
        return (partitions, None)
    if any((partition._length_cache is None for row in partitions for partition in row)):
        chunk_size = compute_chunksize(num_existing_partitions, ideal_num_new_partitions, min_block_size=1)
        new_partitions = np.array([cls.column_partitions(partitions[i:i + chunk_size], full_axis=False) for i in range(0, num_existing_partitions, chunk_size)])
        return (new_partitions, None)
    new_partitions = []
    start = 0
    total_rows = sum((part.length() for part in partitions[:, 0]))
    ideal_partition_size = compute_chunksize(total_rows, ideal_num_new_partitions, min_block_size=1)
    for _ in range(ideal_num_new_partitions):
        if start >= len(partitions):
            break
        stop = start
        partition_size = partitions[start][0].length()
        while stop < len(partitions) and partition_size < ideal_partition_size:
            stop += 1
            if stop < len(partitions):
                partition_size += partitions[stop][0].length()
        if partition_size > ideal_partition_size * max_excess_of_num_partitions:
            prev_length = sum((row[0].length() for row in partitions[start:stop]))
            new_last_partition_size = ideal_partition_size - prev_length
            partitions = np.insert(partitions, stop + 1, [obj.mask(slice(new_last_partition_size, None), slice(None)) for obj in partitions[stop]], 0)
            for obj in partitions[stop + 1]:
                obj._length_cache = partition_size - (prev_length + new_last_partition_size)
            partitions[stop, :] = [obj.mask(slice(None, new_last_partition_size), slice(None)) for obj in partitions[stop]]
            for obj in partitions[stop]:
                obj._length_cache = new_last_partition_size
        new_partitions.append(cls.column_partitions(partitions[start:stop + 1], full_axis=False))
        start = stop + 1
    new_partitions = np.array(new_partitions)
    lengths = [part.length() for part in new_partitions[:, 0]]
    return (new_partitions, lengths)