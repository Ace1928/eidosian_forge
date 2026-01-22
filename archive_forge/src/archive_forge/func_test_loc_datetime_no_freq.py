from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_loc_datetime_no_freq():
    datetime_index = pd.date_range('2016-01-01', '2016-01-31', freq='12h')
    datetime_index.freq = None
    df = pd.DataFrame({'num': range(len(datetime_index))}, index=datetime_index)
    ddf = dd.from_pandas(df, npartitions=1)
    slice_ = slice('2016-01-03', '2016-01-05')
    result = ddf.loc[slice_, :]
    expected = df.loc[slice_, :]
    assert_eq(result, expected)