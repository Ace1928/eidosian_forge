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
def test_apply_warns():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40]})
    ddf = dd.from_pandas(df, npartitions=2)
    func = lambda row: row['x'] + row['y']
    with pytest.warns(UserWarning) as w:
        ddf.apply(func, axis=1)
    assert len(w) == 1
    with warnings.catch_warnings(record=True) as record:
        ddf.apply(func, axis=1, meta=(None, int))
    assert not record
    with pytest.warns(UserWarning) as w:
        ddf.apply(lambda x: x, axis=1)
    assert len(w) == 1
    assert "'x'" in str(w[0].message)
    assert 'int64' in str(w[0].message)