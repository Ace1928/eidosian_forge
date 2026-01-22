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
def test_hdf_nonpandas_keys():
    tables = pytest.importorskip('tables')
    import tables

    class Table1(tables.IsDescription):
        value1 = tables.Float32Col()

    class Table2(tables.IsDescription):
        value2 = tables.Float32Col()

    class Table3(tables.IsDescription):
        value3 = tables.Float32Col()
    with tmpfile('h5') as path:
        with tables.open_file(path, mode='a') as h5file:
            group = h5file.create_group('/', 'group')
            t = h5file.create_table(group, 'table1', Table1, 'Table 1')
            row = t.row
            row['value1'] = 1
            row.append()
            t = h5file.create_table(group, 'table2', Table2, 'Table 2')
            row = t.row
            row['value2'] = 1
            row.append()
            t = h5file.create_table(group, 'table3', Table3, 'Table 3')
            row = t.row
            row['value3'] = 1
            row.append()
        bar = pd.DataFrame(np.random.randn(10, 4))
        bar.to_hdf(path, key='/bar', format='table', mode='a')
        dd.read_hdf(path, '/group/table1')
        dd.read_hdf(path, '/group/table2')
        dd.read_hdf(path, '/group/table3')
        dd.read_hdf(path, '/bar')