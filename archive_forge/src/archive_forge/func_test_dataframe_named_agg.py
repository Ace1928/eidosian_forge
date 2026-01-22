from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(not PANDAS_GE_140, reason='requires pandas >= 1.4.0')
@pytest.mark.parametrize('shuffle_method', [True, False])
def test_dataframe_named_agg(shuffle_method):
    df = pd.DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 5, 6], 'c': [6, 3, 6, 7]})
    ddf = dd.from_pandas(df, npartitions=2)
    expected = df.groupby('a').agg(x=pd.NamedAgg('b', aggfunc='sum'), y=pd.NamedAgg('c', aggfunc=partial(np.std, ddof=1)))
    actual = ddf.groupby('a').agg(shuffle_method=shuffle_method, x=pd.NamedAgg('b', aggfunc='sum'), y=pd.NamedAgg('c', aggfunc=partial(np.std, ddof=1)))
    assert_eq(expected, actual)