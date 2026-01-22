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
def describe_numeric_aggregate(stats, name=None, is_timedelta_col=False, is_datetime_col=False):
    assert len(stats) == 6
    count, mean, std, min, q, max = stats
    if is_series_like(count):
        typ = type(count.to_frame())
    else:
        typ = type(q)
    if is_timedelta_col:
        mean = pd.to_timedelta(mean)
        std = pd.to_timedelta(std)
        min = pd.to_timedelta(min)
        max = pd.to_timedelta(max)
        q = q.apply(lambda x: pd.to_timedelta(x))
    if is_datetime_col:
        min = pd.to_datetime(min)
        max = pd.to_datetime(max)
        q = q.apply(lambda x: pd.to_datetime(x))
    if is_datetime_col:
        part1 = typ([count, min], index=['count', 'min'])
    else:
        part1 = typ([count, mean, std, min], index=['count', 'mean', 'std', 'min'])
    q.index = [f'{l * 100:g}%' for l in tolist(q.index)]
    if is_series_like(q) and typ != type(q):
        q = q.to_frame()
    part3 = typ([max], index=['max'])
    result = concat([part1, q, part3], sort=False)
    if is_series_like(result):
        result.name = name
    return result