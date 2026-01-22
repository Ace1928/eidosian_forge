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
@pytest.mark.parametrize('use_names', [True, False])
def test_names_with_header_0(tmpdir, use_names):
    csv = StringIO('    city1,1992-09-13,10\n    city2,1992-09-13,14\n    city3,1992-09-13,98\n    city4,1992-09-13,13\n    city5,1992-09-13,45\n    city6,1992-09-13,64\n    ')
    if use_names:
        names = ['city', 'date', 'sales']
        usecols = ['city', 'sales']
    else:
        names = usecols = None
    path = os.path.join(str(tmpdir), 'input.csv')
    pd.read_csv(csv, header=None).to_csv(path, index=False, header=False)
    df = pd.read_csv(path, header=0, names=names, usecols=usecols)
    ddf = dd.read_csv(path, header=0, names=names, usecols=usecols, blocksize=60)
    assert_eq(df, ddf, check_index=False)