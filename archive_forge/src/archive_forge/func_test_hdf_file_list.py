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
def test_hdf_file_list():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    with tmpdir() as tdir:
        df.iloc[:2].to_hdf(os.path.join(tdir, 'one.h5'), key='dataframe', format='table')
        df.iloc[2:].to_hdf(os.path.join(tdir, 'two.h5'), key='dataframe', format='table')
        with dask.config.set(scheduler='sync'):
            input_files = [os.path.join(tdir, 'one.h5'), os.path.join(tdir, 'two.h5')]
            res = dd.read_hdf(input_files, 'dataframe')
            tm.assert_frame_equal(res.compute(), df)