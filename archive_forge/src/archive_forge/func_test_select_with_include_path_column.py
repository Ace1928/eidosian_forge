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
def test_select_with_include_path_column(tmpdir):
    d = {'col1': [i for i in range(0, 100)], 'col2': [i for i in range(100, 200)]}
    df = pd.DataFrame(data=d)
    temp_path = str(tmpdir) + '/'
    for i in range(6):
        df.to_csv(f'{temp_path}file_{i}.csv', index=False)
    ddf = dd.read_csv(temp_path + '*.csv', include_path_column=True)
    assert_eq(ddf.col1, pd.concat([df.col1] * 6))