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
def test_loc_with_non_boolean_series():
    df = pd.Series(np.random.randn(20), index=list('abcdefghijklmnopqrst'))
    ddf = dd.from_pandas(df, 3)
    s = pd.Series(list('bdmnat'))
    ds = dd.from_pandas(s, npartitions=3)
    msg = 'Cannot index with non-boolean dask Series. Try passing computed values instead'
    with pytest.raises(KeyError, match=msg):
        ddf.loc[ds]
    assert_eq(ddf.loc[s], df.loc[s])
    ctx = contextlib.nullcontext()
    if pyarrow_strings_enabled():
        ctx = pytest.warns(UserWarning, match='converting pandas extension dtypes to arrays')
    with ctx:
        with pytest.raises(KeyError, match=msg):
            ddf.loc[ds.values]
    assert_eq(ddf.loc[s.values], df.loc[s])
    ddf = ddf.clear_divisions()
    with pytest.raises(KeyError, match=msg):
        ddf.loc[ds]
    with pytest.raises(KeyError, match='Cannot index with list against unknown division'):
        ddf.loc[s]