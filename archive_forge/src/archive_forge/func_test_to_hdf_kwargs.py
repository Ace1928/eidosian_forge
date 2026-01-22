from __future__ import annotations
import os
import pathlib
from time import sleep
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.dataframe._compat import tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
from dask.layers import DataFrameIOLayer
from dask.utils import dependency_depth, tmpdir, tmpfile
def test_to_hdf_kwargs():
    pytest.importorskip('tables')
    df = pd.DataFrame({'A': ['a', 'aaaa']})
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpfile('h5') as fn:
        ddf.to_hdf(fn, 'foo4', format='table', min_itemsize=4)
        df2 = pd.read_hdf(fn, 'foo4')
        tm.assert_frame_equal(df, df2)
    with tmpfile('h5') as fn:
        ddf.to_hdf(fn, 'foo4', format='t', min_itemsize=4)
        df2 = pd.read_hdf(fn, 'foo4')
        tm.assert_frame_equal(df, df2)