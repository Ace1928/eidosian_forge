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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='FIXME hanging - this is a bug')
def test_cumulative_multiple_columns():
    df = pd.DataFrame(np.random.randn(100, 5), columns=list('abcde'))
    ddf = dd.from_pandas(df, 5)
    for d in [ddf, df]:
        for c in df.columns:
            d[c + 'cs'] = d[c].cumsum()
            d[c + 'cmin'] = d[c].cummin()
            d[c + 'cmax'] = d[c].cummax()
            d[c + 'cp'] = d[c].cumprod()
    assert_eq(ddf, df)