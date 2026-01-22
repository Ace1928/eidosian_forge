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
@requires_dask
def test_apply_dask_new_output_sizes_not_supplied_same_dim_names() -> None:
    data = np.random.randn(4, 4, 3, 2)
    da = xr.DataArray(data=data, dims=('x', 'y', 'i', 'j')).chunk(x=1, y=1)
    with pytest.raises(ValueError, match='output_sizes'):
        xr.apply_ufunc(np.linalg.pinv, da, input_core_dims=[['i', 'j']], output_core_dims=[['i', 'j']], exclude_dims=set(('i', 'j')), dask='parallelized')