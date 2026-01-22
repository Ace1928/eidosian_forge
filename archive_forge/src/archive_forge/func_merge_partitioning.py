from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def merge_partitioning(left, right, axis=1):
    """
    Get the number of splits across the `axis` for the two dataframes being concatenated.

    Parameters
    ----------
    left : PandasDataframe
    right : PandasDataframe
    axis : int, default: 1

    Returns
    -------
    int
    """
    lshape = left._row_lengths_cache if axis == 0 else left._column_widths_cache
    rshape = right._row_lengths_cache if axis == 0 else right._column_widths_cache
    if lshape is not None and rshape is not None:
        res_shape = sum(lshape) + sum(rshape)
        chunk_size = compute_chunksize(axis_len=res_shape, num_splits=NPartitions.get(), min_block_size=MinPartitionSize.get())
        return ceil(res_shape / chunk_size)
    else:
        lsplits = left._partitions.shape[axis]
        rsplits = right._partitions.shape[axis]
        return min(lsplits + rsplits, NPartitions.get())