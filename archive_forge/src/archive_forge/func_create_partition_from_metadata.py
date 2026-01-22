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
def create_partition_from_metadata(cls, **metadata):
    """
        Create NumPy array of partitions that holds an empty dataframe with given metadata.

        Parameters
        ----------
        **metadata : dict
            Metadata that has to be wrapped in a partition.

        Returns
        -------
        np.ndarray
            A NumPy 2D array of a single partition which contains the data.
        """
    metadata_dataframe = pandas.DataFrame(**metadata)
    return np.array([[cls._partition_class.put(metadata_dataframe)]])