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
@pytest.mark.parametrize('blocksize', [5, 15, 33, 200, 90000])
def test_read_json_multiple_files_with_path_column(blocksize, tmpdir):
    fil1 = str(tmpdir.join('fil1.json')).replace(os.sep, '/')
    fil2 = str(tmpdir.join('fil2.json')).replace(os.sep, '/')
    df = pd.DataFrame({'x': range(5), 'y': ['a', 'b', 'c', 'd', 'e']})
    df2 = df.assign(x=df.x + 0.5)
    orient = 'records'
    lines = True
    df.to_json(fil1, orient=orient, lines=lines)
    df2.to_json(fil2, orient=orient, lines=lines)
    path_dtype = pd.CategoricalDtype((fil1, fil2))
    df['path'] = pd.Series((fil1,) * len(df), dtype=path_dtype)
    df2['path'] = pd.Series((fil2,) * len(df2), dtype=path_dtype)
    sol = pd.concat([df, df2])
    res = dd.read_json(str(tmpdir.join('fil*.json')), orient=orient, lines=lines, include_path_column=True, blocksize=blocksize)
    assert_eq(res, sol, check_index=False)