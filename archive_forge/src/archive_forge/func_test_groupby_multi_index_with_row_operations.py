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
@pytest.mark.parametrize('operation', ['head', 'tail'])
def test_groupby_multi_index_with_row_operations(operation):
    df = pd.DataFrame(data=[['a0', 'b1'], ['a0', 'b2'], ['a1', 'b1'], ['a3', 'b3'], ['a3', 'b3'], ['a5', 'b5'], ['a1', 'b1'], ['a1', 'b1'], ['a1', 'b1']], columns=['A', 'B'])
    caller = operator.methodcaller(operation)
    expected = caller(df.groupby(['A', df['A'].eq('a1')])['B'])
    ddf = dd.from_pandas(df, npartitions=3)
    actual = caller(ddf.groupby(['A', ddf['A'].eq('a1')])['B'])
    assert_eq(expected, actual)