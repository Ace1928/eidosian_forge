from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_alignment_error(self):
    ds = xr.Dataset(coords={'x': [1, 2]})
    other = xr.Dataset(coords={'x': [2, 3]})
    with pytest.raises(ValueError, match='cannot align.*join.*exact.*not equal.*'):
        xr.merge([ds, other], join='exact')