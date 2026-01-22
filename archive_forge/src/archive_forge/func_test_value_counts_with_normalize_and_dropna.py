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
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_with_normalize_and_dropna(normalize):
    df = pd.DataFrame({'x': [1, 2, 1, 3, np.nan, 1, 4]})
    ddf = dd.from_pandas(df, npartitions=3)
    result = ddf.x.value_counts(dropna=False, normalize=normalize)
    expected = df.x.value_counts(dropna=False, normalize=normalize)
    assert_eq(result, expected)
    result2 = ddf.x.value_counts(split_every=2, dropna=False, normalize=normalize)
    assert_eq(result2, expected)
    assert result._name != result2._name
    result3 = ddf.x.value_counts(split_out=2, dropna=False, normalize=normalize)
    assert_eq(result3, expected)
    assert result._name != result3._name
    result4 = ddf.x.value_counts(dropna=True, normalize=normalize, split_out=2)
    expected4 = df.x.value_counts(dropna=True, normalize=normalize)
    assert_eq(result4, expected4)