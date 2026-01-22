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
def test_missing_values(self):
    a = np.array([0, 1, np.nan, 3])
    s = sparse.COO.from_numpy(a)
    var_s = Variable('x', s)
    assert np.all(var_s.fillna(2).data.todense() == np.arange(4))
    assert np.all(var_s.count() == 3)