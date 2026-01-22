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
@pytest.mark.parametrize('header,header_first_partition_only,expected_first,expected_next', [(False, False, 'a,1\n', 'd,4\n'), (True, False, 'x,y\n', 'x,y\n'), (False, True, 'a,1\n', 'd,4\n'), (True, True, 'x,y\n', 'd,4\n'), (['aa', 'bb'], False, 'aa,bb\n', 'aa,bb\n'), (['aa', 'bb'], True, 'aa,bb\n', 'd,4\n')])
def test_to_csv_header(header, header_first_partition_only, expected_first, expected_next):
    partition_count = 2
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f'], 'y': [1, 2, 3, 4, 5, 6]})
    ddf = dd.from_pandas(df, npartitions=partition_count)
    with tmpdir() as dn:
        ddf.to_csv(os.path.join(dn, 'fooa*.csv'), index=False, header=header, header_first_partition_only=header_first_partition_only)
        filename = os.path.join(dn, 'fooa0.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected_first
        os.remove(filename)
        filename = os.path.join(dn, 'fooa1.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected_next
        os.remove(filename)