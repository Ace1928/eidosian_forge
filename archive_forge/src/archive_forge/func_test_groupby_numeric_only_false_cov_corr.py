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
@pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not supported for <1.5')
@pytest.mark.parametrize('func', ['cov', 'corr'])
def test_groupby_numeric_only_false_cov_corr(func):
    df = pd.DataFrame({'float': [1.0, 2.0, 3.0, 4.0, 5, 6.0, 7.0, 8.0], 'int': [1, 2, 3, 4, 5, 6, 7, 8], 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, 8]), 'A': 1})
    ddf = dd.from_pandas(df, npartitions=2)
    dd_result = getattr(ddf.groupby('A'), func)(numeric_only=False)
    pd_result = getattr(df.groupby('A'), func)(numeric_only=False)
    assert_eq(dd_result, pd_result)
    dd_result = getattr(ddf.groupby('A'), func)(numeric_only=True)
    pd_result = getattr(df.groupby('A'), func)(numeric_only=True)
    assert_eq(dd_result, pd_result)