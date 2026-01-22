from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
def test_to_csv():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpdir() as dn:
            a.to_csv(dn, index=False)
            result = dd.read_csv(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)
        with tmpdir() as dn:
            r = a.to_csv(dn, index=False, compute=False)
            paths = dask.compute(*r, scheduler='sync')
            assert paths == tuple((os.path.join(dn, f'{n}.part') for n in range(npartitions)))
            result = dd.read_csv(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)
        with tmpdir() as dn:
            fn = os.path.join(dn, 'data_*.csv')
            paths = a.to_csv(fn, index=False)
            assert paths == [os.path.join(dn, f'data_{n}.csv') for n in range(npartitions)]
            result = dd.read_csv(fn).compute().reset_index(drop=True)
            assert_eq(result, df)