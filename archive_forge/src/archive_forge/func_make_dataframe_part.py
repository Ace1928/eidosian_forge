from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def make_dataframe_part(index_dtype, start, end, dtypes, columns, state_data, kwargs):
    state = np.random.RandomState(state_data)
    if pd.api.types.is_datetime64_any_dtype(index_dtype):
        index = pd.date_range(start=start, end=end, freq=kwargs.get('freq'), name='timestamp')
    elif pd.api.types.is_integer_dtype(index_dtype):
        step = kwargs.get('freq')
        index = pd.RangeIndex(start=start, stop=end + step, step=step).astype(index_dtype)
    else:
        raise TypeError(f'Unhandled index dtype: {index_dtype}')
    df = make_partition(columns, dtypes, index, kwargs, state)
    while df.index[-1] >= end:
        df = df.iloc[:-1]
    return df