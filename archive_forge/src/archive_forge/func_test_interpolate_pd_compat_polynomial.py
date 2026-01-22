from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
def test_interpolate_pd_compat_polynomial():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    orders = [1, 2, 3]
    for shape, frac_nan, order in itertools.product(shapes, frac_nans, orders):
        da, df = make_interpolate_example_data(shape, frac_nan)
        for dim in ['time', 'x']:
            actual = da.interpolate_na(method='polynomial', order=order, dim=dim, use_coordinate=False)
            expected = df.interpolate(method='polynomial', order=order, axis=da.get_axis_num(dim))
            np.testing.assert_allclose(actual.values, expected.values)