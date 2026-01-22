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
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_bivariate_ufunc(self):
    assert_sparse_equal(np.maximum(self.data, 0), np.maximum(self.var, 0).data)
    assert_sparse_equal(np.maximum(self.data, 0), np.maximum(0, self.var).data)