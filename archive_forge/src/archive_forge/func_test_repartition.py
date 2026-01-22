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
def test_repartition():

    def _check_split_data(orig, d):
        """Check data is split properly"""
        if DASK_EXPR_ENABLED:
            return
        if d is orig:
            return
        keys = [k for k in d.dask if k[0].startswith('repartition-split')]
        keys = sorted(keys)
        sp = pd.concat([compute_as_if_collection(dd.DataFrame, d.dask, k) for k in keys])
        assert_eq(orig, sp)
        assert_eq(orig, d)
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': list('abdabd')}, index=[10, 20, 30, 40, 50, 60])
    a = dd.from_pandas(df, 2)
    b = a.repartition(divisions=[10, 20, 50, 60])
    assert b.divisions == (10, 20, 50, 60)
    assert_eq(a, b)
    if not DASK_EXPR_ENABLED:
        assert_eq(compute_as_if_collection(dd.DataFrame, b.dask, (b._name, 0)), df.iloc[:1])
    for div in [[20, 60], [10, 50], [1], [0, 60], [10, 70], [10, 50, 20, 60], [10, 10, 20, 60]]:
        pytest.raises(ValueError, lambda div=div: a.repartition(divisions=div).compute())
    pdf = pd.DataFrame(np.random.randn(7, 5), columns=list('abxyz'))
    ps = pdf.x
    for p in range(1, 7):
        ddf = dd.from_pandas(pdf, p)
        ds = ddf.x
        assert_eq(ddf, pdf)
        assert_eq(ps, ds)
        for div in [[0, 6], [0, 6, 6], [0, 5, 6], [0, 4, 6, 6], [0, 2, 6], [0, 2, 6, 6], [0, 2, 3, 6, 6], [0, 1, 2, 3, 4, 5, 6, 6]]:
            rddf = ddf.repartition(divisions=div)
            _check_split_data(ddf, rddf)
            assert rddf.divisions == tuple(div)
            assert_eq(pdf, rddf)
            rds = ds.repartition(divisions=div)
            _check_split_data(ds, rds)
            assert rds.divisions == tuple(div)
            assert_eq(pdf.x, rds)
        for div in [[-5, 10], [-2, 3, 5, 6], [0, 4, 5, 9, 10]]:
            rddf = ddf.repartition(divisions=div, force=True)
            _check_split_data(ddf, rddf)
            assert rddf.divisions == tuple(div)
            assert_eq(pdf, rddf)
            rds = ds.repartition(divisions=div, force=True)
            _check_split_data(ds, rds)
            assert rds.divisions == tuple(div)
            assert_eq(pdf.x, rds)
    pdf = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}, index=list('abcdefghij'))
    ps = pdf.x
    for p in range(1, 7):
        ddf = dd.from_pandas(pdf, p)
        ds = ddf.x
        assert_eq(ddf, pdf)
        assert_eq(ps, ds)
        for div in [list('aj'), list('ajj'), list('adj'), list('abfj'), list('ahjj'), list('acdj'), list('adfij'), list('abdefgij'), list('abcdefghij')]:
            rddf = ddf.repartition(divisions=div)
            _check_split_data(ddf, rddf)
            assert rddf.divisions == tuple(div)
            assert_eq(pdf, rddf)
            rds = ds.repartition(divisions=div)
            _check_split_data(ds, rds)
            assert rds.divisions == tuple(div)
            assert_eq(pdf.x, rds)
        for div in [list('Yadijm'), list('acmrxz'), list('Yajz')]:
            rddf = ddf.repartition(divisions=div, force=True)
            _check_split_data(ddf, rddf)
            assert rddf.divisions == tuple(div)
            assert_eq(pdf, rddf)
            rds = ds.repartition(divisions=div, force=True)
            _check_split_data(ds, rds)
            assert rds.divisions == tuple(div)
            assert_eq(pdf.x, rds)