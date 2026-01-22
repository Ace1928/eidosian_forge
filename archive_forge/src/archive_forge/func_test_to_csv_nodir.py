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
@pytest.mark.skipif(Version(fsspec.__version__) == Version('2023.9.1'), reason='https://github.com/dask/dask/issues/10515')
def test_to_csv_nodir():
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir0 = os.path.join(str(dir), 'createme')
        df.to_csv(dir0)
        assert 'createme' in os.listdir(dir)
        assert os.listdir(dir0)
        result = dd.read_csv(os.path.join(dir0, '*')).compute()
    assert (result.x.values == df0.x.values).all()