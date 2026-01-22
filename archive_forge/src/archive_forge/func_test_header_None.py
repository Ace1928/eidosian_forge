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
def test_header_None():
    with filetexts({'.tmp.1.csv': '1,2', '.tmp.2.csv': '', '.tmp.3.csv': '3,4'}):
        df = dd.read_csv('.tmp.*.csv', header=None)
        expected = pd.DataFrame({0: [1, 3], 1: [2, 4]})
        assert_eq(df.compute().reset_index(drop=True), expected)