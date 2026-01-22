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
def wait_computations_if_benchmark_mode(func):
    """
    Make sure a `func` finished its computations in benchmark mode.

    Parameters
    ----------
    func : callable
        A function that should be performed in syncronous mode.

    Returns
    -------
    callable
        Wrapped function that executes eagerly (if benchmark mode) or original `func`.

    Notes
    -----
    `func` should return NumPy array with partitions.
    """

    @wraps(func)
    def wait(cls, *args, **kwargs):
        """Wait for computation results."""
        result = func(cls, *args, **kwargs)
        if BenchmarkMode.get():
            if isinstance(result, tuple):
                partitions = result[0]
            else:
                partitions = result
            cls.finalize(partitions)
            cls.wait_partitions(partitions.flatten())
        return result
    return wait