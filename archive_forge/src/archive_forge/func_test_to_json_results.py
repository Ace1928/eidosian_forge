from __future__ import annotations
import json
import os
import fsspec
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200
from dask.dataframe.utils import assert_eq
from dask.utils import tmpdir, tmpfile
@pytest.mark.skipif(Version(fsspec.__version__) == Version('2023.9.1'), reason='https://github.com/dask/dask/issues/10515')
def test_to_json_results():
    with tmpfile('json') as f:
        paths = ddf.to_json(f)
        assert paths == [os.path.join(f, f'{n}.part') for n in range(ddf.npartitions)]
    with tmpfile('json') as f:
        list_of_delayed = ddf.to_json(f, compute=False)
        paths = dask.compute(*list_of_delayed)
        assert paths == tuple((os.path.join(f, f'{n}.part') for n in range(ddf.npartitions)))