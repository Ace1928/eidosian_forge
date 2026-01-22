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
@pytest.mark.parametrize('blocksize', [None, 10])
@pytest.mark.parametrize('fmt', compression_fmts)
def test_read_csv_compression(fmt, blocksize):
    if fmt and fmt not in compress:
        pytest.skip('compress function not provided for %s' % fmt)
    expected = read_files()
    suffix = {'gzip': '.gz', 'bz2': '.bz2', 'zip': '.zip', 'xz': '.xz'}.get(fmt, '')
    files2 = valmap(compress[fmt], csv_files) if fmt else csv_files
    renamed_files = {k + suffix: v for k, v in files2.items()}
    with filetexts(renamed_files, mode='b'):
        if fmt and blocksize:
            with pytest.warns(UserWarning):
                df = dd.read_csv('2014-01-*.csv' + suffix, blocksize=blocksize)
        else:
            df = dd.read_csv('2014-01-*.csv' + suffix, blocksize=blocksize)
        assert_eq(df.compute(scheduler='sync').reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)