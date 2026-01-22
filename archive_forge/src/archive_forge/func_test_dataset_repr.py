from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_dataset_repr(self):
    ds = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}, coords={'y': ('x', sparse.COO.from_numpy(np.arange(4, dtype='i8')))})
    expected = dedent('            <xarray.Dataset> Size: 112B\n            Dimensions:  (x: 4)\n            Coordinates:\n                y        (x) int64 48B <COO: nnz=3, fill_value=0>\n            Dimensions without coordinates: x\n            Data variables:\n                a        (x) float64 64B <COO: nnz=4, fill_value=0.0>')
    assert expected == repr(ds)