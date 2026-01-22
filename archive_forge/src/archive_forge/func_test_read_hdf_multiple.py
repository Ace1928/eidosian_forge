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
@pytest.mark.skipif(PY_VERSION >= Version('3.11'), reason='segfaults due to https://github.com/PyTables/PyTables/issues/977')
def test_read_hdf_multiple():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}, index=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    a = dd.from_pandas(df, 16)
    with tmpfile('h5') as fn:
        a.to_hdf(fn, '/data*')
        r = dd.read_hdf(fn, '/data*', sorted_index=True)
        assert a.npartitions == r.npartitions
        assert a.divisions == r.divisions
        assert_eq(a, r)