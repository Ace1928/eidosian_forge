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
def make_timeseries(start='2000-01-01', end='2000-12-31', dtypes=None, freq='10s', partition_freq=f'1{_ME}', seed=None, **kwargs):
    """Create timeseries dataframe with random data

    Parameters
    ----------
    start: datetime (or datetime-like string)
        Start of time series
    end: datetime (or datetime-like string)
        End of time series
    dtypes: dict (optional)
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}
    freq: string
        String like '2s' or '1H' or '12W' for the time series frequency
    partition_freq: string
        String like '1M' or '2Y' to divide the dataframe into partitions
    seed: int (optional)
        Randomstate seed
    kwargs:
        Keywords to pass down to individual column creation functions.
        Keywords should be prefixed by the column name and then an underscore.

    Examples
    --------
    >>> import dask.dataframe as dd
    >>> df = dd.demo.make_timeseries('2000', '2010',
    ...                              {'value': float, 'name': str, 'id': int},
    ...                              freq='2h', partition_freq='1D', seed=1)
    >>> df.head()  # doctest: +SKIP
                           id      name     value
    2000-01-01 00:00:00   969     Jerry -0.309014
    2000-01-01 02:00:00  1010       Ray -0.760675
    2000-01-01 04:00:00  1016  Patricia -0.063261
    2000-01-01 06:00:00   960   Charlie  0.788245
    2000-01-01 08:00:00  1031     Kevin  0.466002
    """
    if dtypes is None:
        dtypes = {'name': str, 'id': int, 'x': float, 'y': float}
    divisions = list(pd.date_range(start=start, end=end, freq=partition_freq))
    npartitions = len(divisions) - 1
    if seed is None:
        state_data = np.random.randint(2000000000.0, size=npartitions)
    else:
        state_data = random_state_data(npartitions, seed)
    parts = []
    for i in range(len(divisions) - 1):
        parts.append((divisions[i:i + 2], state_data[i]))
    kwargs['freq'] = freq
    index_dtype = 'datetime64[ns]'
    meta_start, meta_end = list(pd.date_range(start='2000', freq=freq, periods=2))
    from dask.dataframe import _dask_expr_enabled
    if _dask_expr_enabled():
        from dask_expr import from_map
        k = {}
    else:
        from dask.dataframe.io.io import from_map
        k = {'token': tokenize(start, end, dtypes, freq, partition_freq, state_data)}
    return from_map(MakeDataframePart(index_dtype, dtypes, kwargs), parts, meta=make_dataframe_part(index_dtype, meta_start, meta_end, dtypes, list(dtypes.keys()), state_data[0], kwargs), divisions=divisions, label='make-timeseries', enforce_metadata=False, **k)