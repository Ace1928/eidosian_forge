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
def test_robust_column_mismatch():
    files = csv_files.copy()
    k = sorted(files)[-1]
    files[k] = files[k].replace(b'name', b'Name')
    with filetexts(files, mode='b'):
        ddf = dd.read_csv('2014-01-*.csv', header=None, skiprows=1, names=['name', 'amount', 'id'])
        df = pd.read_csv('2014-01-01.csv')
        assert (df.columns == ddf.columns).all()
        assert_eq(ddf, ddf)