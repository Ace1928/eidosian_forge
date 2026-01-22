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
@pytest.mark.skip
def test_warn_non_seekable_files():
    files2 = valmap(compress['gzip'], csv_files)
    with filetexts(files2, mode='b'):
        with pytest.warns(UserWarning) as w:
            df = dd.read_csv('2014-01-*.csv', compression='gzip')
            assert df.npartitions == 3
        assert len(w) == 1
        msg = str(w[0].message)
        assert 'gzip' in msg
        assert 'blocksize=None' in msg
        with warnings.catch_warnings(record=True) as record:
            df = dd.read_csv('2014-01-*.csv', compression='gzip', blocksize=None)
        assert not record
        with pytest.raises(NotImplementedError):
            with pytest.warns(UserWarning):
                df = dd.read_csv('2014-01-*.csv', compression='foo')