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
def test_read_json_path_column_with_duplicate_name_is_error():
    with tmpfile('json') as f:
        df.to_json(f, orient='records', lines=False)
        with pytest.raises(ValueError, match='Files already contain'):
            dd.read_json(f, orient='records', lines=False, include_path_column='x')