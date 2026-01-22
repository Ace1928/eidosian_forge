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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
def test_dask_layers():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8]})
    ddf = dd.from_pandas(df, npartitions=2)
    assert ddf.dask.layers.keys() == {ddf._name}
    assert ddf.dask.dependencies == {ddf._name: set()}
    assert ddf.__dask_layers__() == (ddf._name,)
    dds = ddf['x']
    assert dds.dask.layers.keys() == {ddf._name, dds._name}
    assert dds.dask.dependencies == {ddf._name: set(), dds._name: {ddf._name}}
    assert dds.__dask_layers__() == (dds._name,)
    ddi = dds.min()
    assert ddi.key[1:] == (0,)
    assert {ddf._name, dds._name, ddi.key[0]}.issubset(ddi.dask.layers.keys())
    assert len(ddi.dask.layers) == 4
    assert ddi.dask.dependencies[ddf._name] == set()
    assert ddi.dask.dependencies[dds._name] == {ddf._name}
    assert len(ddi.dask.dependencies) == 4
    assert ddi.__dask_layers__() == (ddi.key[0],)