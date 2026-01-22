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
def test_to_csv_line_ending():
    df = pd.DataFrame({'x': [0]})
    ddf = dd.from_pandas(df, npartitions=1)
    expected = {b'0\r\n', b'0\n'}
    with tmpdir() as dn:
        ddf.to_csv(os.path.join(dn, 'foo*.csv'), header=False, index=False)
        filename = os.path.join(dn, 'foo0.csv')
        with open(filename, 'rb') as f:
            raw = f.read()
    assert raw in expected