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
@pytest.mark.slow
def test_to_hdf_lock_delays():
    pytest.importorskip('tables')
    df16 = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}, index=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    a = dd.from_pandas(df16, 16)

    def delayed_nop(i):
        if i.iloc[1] < 10:
            sleep(0.1 * (10 - i.iloc[1]))
        return i
    with tmpfile() as fn:
        a = a.apply(delayed_nop, axis=1, meta=a)
        a.to_hdf(fn, '/data*')
        out = dd.read_hdf(fn, '/data*')
        assert_eq(df16, out)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data*')
        a = a.apply(delayed_nop, axis=1, meta=a)
        a.to_hdf(fn, '/data')
        out = dd.read_hdf(fn, '/data')
        assert_eq(df16, out)