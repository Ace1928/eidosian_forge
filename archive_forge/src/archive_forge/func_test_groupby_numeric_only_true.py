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
@pytest.mark.parametrize('func', ['min', 'max', 'sum', 'prod', 'first', 'last', 'corr', 'cov', 'cumprod', 'cumsum', 'mean', 'median', 'std', 'var'])
def test_groupby_numeric_only_true(func):
    df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [3, 4, 3, 4], 'C': ['a', 'b', 'c', 'd']})
    ddf = dd.from_pandas(df, npartitions=2)
    if func in ['var', 'std', 'cov', 'corr'] and (not PANDAS_GE_150):
        with pytest.raises(TypeError, match='numeric_only not supported'):
            getattr(ddf.groupby('A'), func)(numeric_only=True)
        with pytest.raises(TypeError, match='got an unexpected keyword'):
            getattr(df.groupby('A'), func)(numeric_only=True)
    else:
        ddf_result = getattr(ddf.groupby('A'), func)(numeric_only=True)
        pdf_result = getattr(df.groupby('A'), func)(numeric_only=True)
        assert_eq(ddf_result, pdf_result)