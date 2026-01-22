from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def split_result_of_axis_func_pandas(axis: int, num_splits: int, result: pandas.DataFrame, min_block_size: int, length_list: Optional[list]=None) -> list[pandas.DataFrame]:
    """
    Split pandas DataFrame evenly based on the provided number of splits.

    Parameters
    ----------
    axis : {0, 1}
        Axis to split across. 0 means index axis when 1 means column axis.
    num_splits : int
        Number of splits to separate the DataFrame into.
        This parameter is ignored if `length_list` is specified.
    result : pandas.DataFrame
        DataFrame to split.
    min_block_size : int
        Minimum number of rows/columns in a single split.
    length_list : list of ints, optional
        List of slice lengths to split DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Returns
    -------
    list of pandas.DataFrames
        Splitted dataframe represented by list of frames.
    """
    return list(generate_result_of_axis_func_pandas(axis, num_splits, result, min_block_size, length_list))