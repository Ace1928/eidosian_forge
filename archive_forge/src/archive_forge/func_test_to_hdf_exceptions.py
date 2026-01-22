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
def test_to_hdf_exceptions():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    a = dd.from_pandas(df, 1)
    with tmpdir() as dn:
        with pytest.raises(ValueError):
            fn = os.path.join(dn, 'data_*.h5')
            a.to_hdf(fn, '/data_*')
    with tmpfile() as fn:
        with pd.HDFStore(fn) as hdf:
            with pytest.raises(ValueError):
                a.to_hdf(hdf, '/data_*_*')