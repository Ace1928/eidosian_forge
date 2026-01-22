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
def split_pandas_df_into_partitions(cls, df, row_chunksize, col_chunksize, update_bar):
    """
        Split given pandas DataFrame according to the row/column chunk sizes into distributed partitions.

        Parameters
        ----------
        df : pandas.DataFrame
        row_chunksize : int
        col_chunksize : int
        update_bar : callable(x) -> x
            Function that updates a progress bar.

        Returns
        -------
        2D np.ndarray[PandasDataframePartition]
        """
    put_func = cls._partition_class.put
    if col_chunksize >= len(df.columns):
        col_parts = [df]
    else:
        col_parts = [df.iloc[:, i:i + col_chunksize] for i in range(0, len(df.columns), col_chunksize)]
    parts = [[update_bar(put_func(col_part.iloc[i:i + row_chunksize])) for col_part in col_parts] for i in range(0, len(df), row_chunksize)]
    return np.array(parts)