from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_error(self):
    ds = xr.Dataset({'x': 0})
    with pytest.raises(xr.MergeError):
        xr.merge([ds, ds + 1])