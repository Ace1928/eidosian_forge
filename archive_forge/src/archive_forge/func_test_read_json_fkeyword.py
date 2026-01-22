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
@pytest.mark.parametrize('fkeyword', ['pandas', 'json'])
def test_read_json_fkeyword(fkeyword):

    def _my_json_reader(*args, **kwargs):
        if fkeyword == 'json':
            return pd.DataFrame.from_dict(json.load(*args))
        return pd.read_json(*args)
    with tmpfile('json') as f:
        df.to_json(f, orient='records', lines=False)
        actual = dd.read_json(f, orient='records', lines=False, engine=_my_json_reader)
        actual_pd = pd.read_json(f, orient='records', lines=False)
        assert_eq(actual, actual_pd)