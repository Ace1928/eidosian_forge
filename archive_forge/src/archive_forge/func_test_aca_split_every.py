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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='Not public')
def test_aca_split_every():
    df = pd.DataFrame({'x': [1] * 60})
    ddf = dd.from_pandas(df, npartitions=15)

    def chunk(x, y, constant=0):
        return x.sum() + y + constant

    def combine(x, constant=0):
        return x.sum() + constant + 1

    def agg(x, constant=0):
        return x.sum() + constant + 2
    f = lambda n: aca([ddf, 2.0], chunk=chunk, aggregate=agg, combine=combine, chunk_kwargs=dict(constant=1.0), combine_kwargs=dict(constant=2.0), aggregate_kwargs=dict(constant=3.0), split_every=n)
    assert_max_deps(f(3), 3)
    assert_max_deps(f(4), 4, False)
    assert_max_deps(f(5), 5)
    assert f(15).dask.keys() == f(ddf.npartitions).dask.keys()
    r3 = f(3)
    r4 = f(4)
    assert r3._name != r4._name
    assert len(r3.dask.keys() & r4.dask.keys()) == len(ddf.dask)
    assert f(3).compute() == 60 + 15 * (2 + 1) + 7 * (2 + 1) + (3 + 2)
    res = aca([ddf, 2.0], chunk=chunk, aggregate=agg, combine=combine, constant=3.0, split_every=3)
    assert res.compute() == 60 + 15 * (2 + 3) + 7 * (3 + 1) + (3 + 2)
    res = aca([ddf, 2.0], chunk=chunk, aggregate=agg, constant=3, split_every=3)
    assert res.compute() == 60 + 15 * (2 + 3) + 8 * (3 + 2)
    with pytest.raises(ValueError):
        f(1)
    with pytest.raises(ValueError):
        aca([ddf, 2.0], chunk=chunk, aggregate=agg, split_every=3, chunk_kwargs=dict(constant=1.0), combine_kwargs=dict(constant=2.0), aggregate_kwargs=dict(constant=3.0))