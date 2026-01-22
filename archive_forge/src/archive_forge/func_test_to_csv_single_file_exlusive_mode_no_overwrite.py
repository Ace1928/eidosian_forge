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
def test_to_csv_single_file_exlusive_mode_no_overwrite():
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as directory:
        csv_path = os.path.join(str(directory), 'test.csv')
        df.to_csv(csv_path, index=False, mode='x', single_file=True)
        assert os.path.exists(csv_path)
        with pytest.raises(FileExistsError):
            df.to_csv(csv_path, index=False, mode='x', single_file=True)
        df.to_csv(csv_path, index=False, mode='w', single_file=True)