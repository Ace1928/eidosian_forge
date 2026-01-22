from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@pytest.mark.parametrize('fill_value', [None, np.nan, 47.11])
@pytest.mark.parametrize('method', ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'])
@requires_scipy
def test_interpolate_pd_compat(method, fill_value) -> None:
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    for shape, frac_nan in itertools.product(shapes, frac_nans):
        da, df = make_interpolate_example_data(shape, frac_nan)
        for dim in ['time', 'x']:
            actual = da.interpolate_na(method=method, dim=dim, fill_value=fill_value)
            expected = df.interpolate(method=method, axis=da.get_axis_num(dim), limit_direction='both', fill_value=fill_value)
            if method == 'linear':
                fixed = expected.values.copy()
                fixed[pd.isnull(actual.values)] = np.nan
                fixed[actual.values == fill_value] = fill_value
            else:
                fixed = expected.values
            np.testing.assert_allclose(actual.values, fixed)