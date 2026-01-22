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
def test_dataarray_pickle(self):
    a1 = xr.DataArray(sparse.COO.from_numpy(np.ones(4)), dims=['x'], coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
    a2 = pickle.loads(pickle.dumps(a1))
    assert_identical(a1, a2)