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
@pytest.mark.parametrize('by', ['A', ['A', 'B']])
def test_empty_partitions_with_value_counts(by):
    df = pd.DataFrame(data=[['a1', 'b1', True], ['a1', None, False], ['a1', 'b1', True], [None, None, None], [None, None, None], [None, None, None], ['a3', 'b3', True], ['a3', 'b3', False], ['a5', 'b5', True]], columns=['A', 'B', 'C'])
    if pyarrow_strings_enabled():
        df = df.convert_dtypes()
    expected = df.groupby(by).C.value_counts()
    ddf = dd.from_pandas(df, npartitions=3)
    actual = ddf.groupby(by).C.value_counts()
    assert_eq(expected, actual)