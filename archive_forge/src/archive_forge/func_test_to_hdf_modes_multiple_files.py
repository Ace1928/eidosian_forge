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
def test_to_hdf_modes_multiple_files():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    a = dd.from_pandas(df, 1)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data*')
        a.to_hdf(os.path.join(dn, 'data2'), '/data')
        a.to_hdf(fn, '/data', mode='a')
        out = dd.read_hdf(fn, '/data*')
        assert_eq(dd.concat([df, df]), out)
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data*')
        a.to_hdf(os.path.join(dn, 'data2'), '/data')
        a.to_hdf(fn, '/data', mode='a')
        out = dd.read_hdf(fn, '/data')
        assert_eq(dd.concat([df, df]), out)
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data*')
        a.to_hdf(os.path.join(dn, 'data1'), '/data')
        a.to_hdf(fn, '/data', mode='w')
        out = dd.read_hdf(fn, '/data')
        assert_eq(df, out)
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data*')
        a.to_hdf(os.path.join(dn, 'data1'), '/data')
        a.to_hdf(fn, '/data', mode='a', append=False)
        out = dd.read_hdf(fn, '/data')
        assert_eq(dd.concat([df, df]), out)