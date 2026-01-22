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
def test_read_json_error():
    with tmpfile('json') as f:
        with pytest.raises(ValueError):
            df.to_json(f, orient='split', lines=True)
        df.to_json(f, orient='split', lines=False)
        with pytest.raises(ValueError):
            dd.read_json(f, orient='split', blocksize=1)