from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_polyval_degree_dim_checks() -> None:
    x = xr.DataArray([1, 2, 3], dims='x')
    coeffs = xr.DataArray([2, 3, 4], dims='degree', coords={'degree': [0, 1, 2]})
    with pytest.raises(ValueError):
        xr.polyval(x, coeffs.drop_vars('degree'))
    with pytest.raises(ValueError):
        xr.polyval(x, coeffs.assign_coords(degree=coeffs.degree.astype(float)))