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
def test_pandas_nullable_boolean_data_type():
    s1 = pd.Series([0, 1, 2])
    s2 = pd.Series([True, False, pd.NA], dtype='boolean')
    ddf1 = dd.from_pandas(s1, npartitions=1)
    ddf2 = dd.from_pandas(s2, npartitions=1)
    assert_eq(ddf1[ddf2], s1[s2])
    assert_eq(ddf1.loc[ddf2], s1.loc[s2])