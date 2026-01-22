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
def fillna_check(df, method, check=True):
    if method:
        out = getattr(df, method)()
    else:
        out = df.fillna()
    if check and out.isnull().values.all(axis=0).any():
        raise ValueError('All NaN partition encountered in `fillna`. Try using ``df.repartition`` to increase the partition size, or specify `limit` in `fillna`.')
    return out