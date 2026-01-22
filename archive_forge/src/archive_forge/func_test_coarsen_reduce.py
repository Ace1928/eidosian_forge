from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
@pytest.mark.slow
@pytest.mark.parametrize('ds', (1, 2), indirect=True)
@pytest.mark.parametrize('window', (1, 2, 3, 4))
@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'var', 'min', 'max', 'median'))
def test_coarsen_reduce(ds: Dataset, window, name) -> None:
    coarsen_obj = ds.coarsen(time=window, boundary='trim')
    actual = coarsen_obj.reduce(getattr(np, f'nan{name}'))
    expected = getattr(coarsen_obj, name)()
    assert_allclose(actual, expected)
    assert list(ds.data_vars.keys()) == list(actual.data_vars.keys())
    for key, src_var in ds.data_vars.items():
        assert src_var.dims == actual[key].dims