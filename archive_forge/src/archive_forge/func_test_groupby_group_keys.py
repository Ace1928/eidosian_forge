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
@pytest.mark.parametrize('group_keys', [True, False, None])
def test_groupby_group_keys(group_keys):
    df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [2, 3, 4, 5]})
    ddf = dd.from_pandas(df, npartitions=2).set_index('a')
    pdf = df.set_index('a')
    func = lambda g: g.copy()
    expected = pdf.groupby('a').apply(func, **INCLUDE_GROUPS)
    assert_eq(expected, ddf.groupby('a').apply(func, meta=expected, **INCLUDE_GROUPS))
    expected = pdf.groupby('a', group_keys=group_keys).apply(func, **INCLUDE_GROUPS)
    assert_eq(expected, ddf.groupby('a', group_keys=group_keys).apply(func, meta=expected, **INCLUDE_GROUPS))