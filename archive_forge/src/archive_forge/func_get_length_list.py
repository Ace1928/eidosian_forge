from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def get_length_list(axis_len: int, num_splits: int, min_block_size: int) -> list:
    """
    Compute partitions lengths along the axis with the specified number of splits.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        Number of splits along the axis.
    min_block_size : int
        Minimum number of rows/columns in a single split.

    Returns
    -------
    list of ints
        List of integer lengths of partitions.
    """
    chunksize = compute_chunksize(axis_len, num_splits, min_block_size)
    return [chunksize if (i + 1) * chunksize <= axis_len else max(0, axis_len - i * chunksize) for i in range(num_splits)]