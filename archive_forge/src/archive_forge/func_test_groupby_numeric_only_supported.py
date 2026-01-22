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
@pytest.mark.parametrize('func', ['min', 'max', 'sum', 'prod', 'first', 'last', 'median', pytest.param('idxmax', marks=pytest.mark.skip(reason='https://github.com/dask/dask/issues/9882')), pytest.param('idxmin', marks=pytest.mark.skip(reason='https://github.com/dask/dask/issues/9882'))])
@pytest.mark.parametrize('numeric_only', [None, True, False])
@pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not implemented for pandas < 1.5')
def test_groupby_numeric_only_supported(func, numeric_only):
    pdf = pd.DataFrame({'ints': [4, 4, 5, 5, 5], 'ints2': [1, 2, 3, 4, 1], 'dates': pd.date_range('2015-01-01', periods=5, freq='1min'), 'strings': ['q', 'c', 'k', 'a', 'l']})
    ddf = dd.from_pandas(pdf, npartitions=3)
    kwargs = {} if numeric_only is None else {'numeric_only': numeric_only}
    ctx = contextlib.nullcontext()
    if PANDAS_GE_150 and (not PANDAS_GE_200):
        if func in ('sum', 'prod', 'median'):
            if numeric_only is None:
                ctx = pytest.warns(FutureWarning, match='The default value of numeric_only')
            elif numeric_only is False:
                ctx = pytest.warns(FutureWarning, match='Dropping invalid columns')
    try:
        with ctx:
            expected = getattr(pdf.groupby('ints'), func)(**kwargs)
        successful_compute = True
    except TypeError:
        ctx = pytest.raises(TypeError, match='Cannot convert|could not convert|does not support|agg function failed')
        successful_compute = False
    with ctx:
        result = getattr(ddf.groupby('ints'), func)(**kwargs)
        if successful_compute:
            assert_eq(expected, result)