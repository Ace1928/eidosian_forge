from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def wrap_skew_reduction(array_skew, index):
    if isinstance(array_skew, np.ndarray) or isinstance(array_skew, list):
        return pd.Series(array_skew, index=index)
    return array_skew