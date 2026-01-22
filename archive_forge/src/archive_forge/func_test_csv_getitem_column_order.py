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
def test_csv_getitem_column_order(tmpdir):
    path = os.path.join(str(tmpdir), 'test.csv')
    columns = list('abcdefghijklmnopqrstuvwxyz')
    values = list(range(len(columns)))
    df1 = pd.DataFrame([{c: v for c, v in zip(columns, values)}])
    df1.to_csv(path)
    columns = list('hczzkylaape')
    df2 = dd.read_csv(path)[columns].head(1)
    assert_eq(df1[columns], df2)