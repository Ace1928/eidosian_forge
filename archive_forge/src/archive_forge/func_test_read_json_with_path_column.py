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
@pytest.mark.parametrize('orient', ['split', 'records', 'index', 'columns', 'values'])
def test_read_json_with_path_column(orient):
    with tmpfile('json') as f:
        df.to_json(f, orient=orient, lines=False)
        actual = dd.read_json(f, orient=orient, lines=False, include_path_column=True)
        actual_pd = pd.read_json(f, orient=orient, lines=False)
        actual_pd['path'] = pd.Series((f.replace(os.sep, '/'),) * len(actual_pd), dtype='category')
        assert actual.path.dtype == 'category'
        assert_eq(actual, actual_pd)