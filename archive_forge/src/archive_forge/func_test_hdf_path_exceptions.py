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
def test_hdf_path_exceptions():
    with pytest.raises(IOError):
        dd.read_hdf('nonexistant_store_X34HJK', '/tmp')
    with pytest.raises(IOError):
        dd.read_hdf(['nonexistant_store_X34HJK', 'nonexistant_store_UY56YH'], '/tmp')
    with pytest.raises(ValueError):
        dd.read_hdf([], '/tmp')