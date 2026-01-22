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
@pytest.mark.xfail
def test_reindex(self):
    x1 = self.ds_xr
    x2 = self.sp_xr
    for kwargs in [{'x': [2, 3, 4]}, {'x': [1, 100, 2, 101, 3]}, {'x': [2.5, 3, 3.5], 'y': [2, 2.5, 3]}]:
        m1 = x1.reindex(**kwargs)
        m2 = x2.reindex(**kwargs)
        assert np.allclose(m1, m2, equal_nan=True)