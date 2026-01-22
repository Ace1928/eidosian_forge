from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.filterwarnings('ignore:In future versions of pandas, numeric_only will be set to False:FutureWarning')
@pytest.mark.parametrize('method,expected', [pytest.param('tdigest', (pd.Series([9.5, 29.5, 19.5], index=['A', 'X', 'B']), pd.DataFrame([[4.5, 24.5, 14.5], [14.5, 34.5, 24.5]], index=[0.25, 0.75], columns=['A', 'X', 'B'])), marks=pytest.mark.skipif(not crick, reason='Requires crick')), ('dask', (pd.Series([7.0, 27.0, 17.0], index=['A', 'X', 'B']), pd.DataFrame([[1.5, 21.5, 11.5], [14.0, 34.0, 24.0]], index=[0.25, 0.75], columns=['A', 'X', 'B'])))])
@pytest.mark.parametrize('numeric_only', [None, True, False])
def test_dataframe_quantile(method, expected, numeric_only):
    df = pd.DataFrame({'A': np.arange(20), 'X': np.arange(20, 40), 'B': np.arange(10, 30), 'C': ['a', 'b', 'c', 'd'] * 5}, columns=['A', 'X', 'B', 'C'])
    ddf = dd.from_pandas(df, 3)
    numeric_only_kwarg = {}
    if numeric_only is not None:
        numeric_only_kwarg = {'numeric_only': numeric_only}
    if numeric_only is False or (PANDAS_GE_200 and numeric_only is None):
        with pytest.raises(TypeError):
            df.quantile(**numeric_only_kwarg)
        with pytest.raises((TypeError, ArrowNotImplementedError, ValueError), match='unsupported operand|no kernel|non-numeric|not supported'):
            ddf.quantile(**numeric_only_kwarg)
    else:
        with assert_numeric_only_default_warning(numeric_only, 'quantile'):
            result = ddf.quantile(method=method, **numeric_only_kwarg)
        assert result.npartitions == 1
        assert result.divisions == ('A', 'X')
        result = result.compute()
        assert isinstance(result, pd.Series)
        assert result.name == 0.5
        assert_eq(result, expected[0], check_names=False)
        with assert_numeric_only_default_warning(numeric_only, 'quantile'):
            result = ddf.quantile([0.25, 0.75], method=method, **numeric_only_kwarg)
        assert result.npartitions == 1
        assert result.divisions == (0.25, 0.75)
        result = result.compute()
        assert isinstance(result, pd.DataFrame)
        tm.assert_index_equal(result.index, pd.Index([0.25, 0.75]))
        tm.assert_index_equal(result.columns, pd.Index(['A', 'X', 'B']))
        assert (result == expected[1]).all().all()
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('ignore', category=FutureWarning)
            expected = df.quantile(axis=1, **numeric_only_kwarg)
        with assert_numeric_only_default_warning(numeric_only, 'quantile'):
            result = ddf.quantile(axis=1, method=method, **numeric_only_kwarg)
        assert_eq(result, expected)
        with pytest.raises(ValueError), assert_numeric_only_default_warning(numeric_only, 'quantile'):
            ddf.quantile([0.25, 0.75], axis=1, method=method, **numeric_only_kwarg)