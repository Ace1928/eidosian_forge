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
def test_Index():
    for case in [pd.DataFrame(np.random.randn(10, 5), index=list('abcdefghij')), pd.DataFrame(np.random.randn(10, 5), index=pd.date_range('2011-01-01', freq='D', periods=10))]:
        ddf = dd.from_pandas(case, 3)
        assert_eq(ddf.index, case.index)
        pytest.raises(AttributeError, lambda ddf=ddf: ddf.index.index)