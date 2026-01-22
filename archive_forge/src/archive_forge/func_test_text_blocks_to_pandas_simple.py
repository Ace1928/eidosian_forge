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
@pytest.mark.skipif(dd._dask_expr_enabled(), reason='not supported')
@csv_and_table
def test_text_blocks_to_pandas_simple(reader, files):
    blocks = [[files[k]] for k in sorted(files)]
    kwargs = {}
    head = pandas_read_text(reader, files['2014-01-01.csv'], b'', {})
    header = files['2014-01-01.csv'].split(b'\n')[0] + b'\n'
    df = text_blocks_to_pandas(reader, blocks, header, head, kwargs)
    assert isinstance(df, dd.DataFrame)
    assert list(df.columns) == ['name', 'amount', 'id']
    values = text_blocks_to_pandas(reader, blocks, header, head, kwargs)
    assert isinstance(values, dd.DataFrame)
    assert hasattr(values, 'dask')
    assert len(values.dask) == 6 if pyarrow_strings_enabled() else 3
    assert_eq(df.amount.sum(), 100 + 200 + 300 + 400 + 500 + 600)