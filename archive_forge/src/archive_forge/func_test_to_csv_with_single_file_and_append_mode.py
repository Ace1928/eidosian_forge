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
def test_to_csv_with_single_file_and_append_mode():
    df0 = pd.DataFrame({'x': ['a', 'b'], 'y': [1, 2]})
    df1 = pd.DataFrame({'x': ['c', 'd'], 'y': [3, 4]})
    df = dd.from_pandas(df1, npartitions=2)
    with tmpdir() as dir:
        csv_path = os.path.join(dir, 'test.csv')
        df0.to_csv(csv_path, index=False)
        df.to_csv(csv_path, mode='a', header=False, index=False, single_file=True)
        result = dd.read_csv(os.path.join(dir, '*')).compute()
    expected = pd.concat([df0, df1])
    assert assert_eq(result, expected, check_index=False)