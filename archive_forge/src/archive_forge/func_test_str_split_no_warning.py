from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.mark.parametrize('index', [None, [0]], ids=['range_index', 'other index'])
def test_str_split_no_warning(index):
    df = pd.DataFrame({'a': ['a\nb']}, index=index)
    ddf = dd.from_pandas(df, npartitions=1)
    pd_a = df['a'].str.split('\n', n=1, expand=True)
    dd_a = ddf['a'].str.split('\n', n=1, expand=True)
    assert_eq(dd_a, pd_a)