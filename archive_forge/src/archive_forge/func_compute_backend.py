from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.datatree import DataTree
from xarray.tests import create_test_data, requires_dask
@pytest.fixture(params=['numbagg', 'bottleneck', None])
def compute_backend(request):
    if request.param is None:
        options = dict(use_bottleneck=False, use_numbagg=False)
    elif request.param == 'bottleneck':
        options = dict(use_bottleneck=True, use_numbagg=False)
    elif request.param == 'numbagg':
        options = dict(use_bottleneck=False, use_numbagg=True)
    else:
        raise ValueError
    with xr.set_options(**options):
        yield request.param