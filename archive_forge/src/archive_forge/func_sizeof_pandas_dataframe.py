from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(pd.DataFrame)
def sizeof_pandas_dataframe(df):
    p = sizeof(df.index) + sizeof(df.columns)
    object_cols = []
    prev_dtype = None
    for col in df._series.values():
        if prev_dtype is None or col.dtype != prev_dtype:
            prev_dtype = col.dtype
            p += 1200
        p += col.memory_usage(index=False, deep=False)
        if col.dtype in OBJECT_DTYPES:
            object_cols.append(col._values)
    p += object_size(*object_cols)
    return max(1200, p)