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
def test_deterministic_hashing_series():
    obj = pd.Series([0, 1, 2])
    dask_df = dd.from_pandas(obj, npartitions=1)
    ddf1 = dask_df.loc[0:1]
    ddf2 = dask_df.loc[0:1]
    if DASK_EXPR_ENABLED:
        assert ddf1._name == ddf2._name
    else:
        assert tokenize(ddf1) == tokenize(ddf2)
    ddf2 = dask_df.loc[0:2]
    if DASK_EXPR_ENABLED:
        assert ddf1._name != ddf2._name
    else:
        assert tokenize(ddf1) != tokenize(ddf2)