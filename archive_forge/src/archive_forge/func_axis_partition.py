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
def axis_partition(cls, partitions, axis, full_axis: bool=True):
    """
        Logically partition along given axis (columns or rows).

        Parameters
        ----------
        partitions : list-like
            List of partitions to be combined.
        axis : {0, 1}
            0 for column partitions, 1 for row partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.
        """
    make_column_partitions = axis == 0
    if not full_axis and (not make_column_partitions):
        raise NotImplementedError("Row partitions must contain the entire axis. We don't " + 'support virtual partitioning for row partitions yet.')
    return cls.column_partitions(partitions) if make_column_partitions else cls.row_partitions(partitions)