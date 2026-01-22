import pickle
import warnings
from unittest import skipIf
import numpy as np
import pandas as pd
import param
import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def assert_apply_xarray(self, expr, expected, skip_dask=False, skip_no_index=False):
    import xarray as xr
    if np.isscalar(expected):
        self.assertEqual(expr.apply(self.dataset_xarray, keep_index=False), expected)
        self.assertEqual(expr.apply(self.dataset_xarray, keep_index=True), expected)
        return
    self.assertIsInstance(expected, xr.DataArray)
    if not skip_no_index:
        np.testing.assert_equal(expr.apply(self.dataset_xarray), expected.values)
    xr.testing.assert_equal(expr.apply(self.dataset_xarray, keep_index=True), expected)
    if skip_dask or da is None:
        return
    expected_da = da.from_array(expected.values)
    expected_dask = expected.copy()
    expected_dask.data = expected_da
    if not skip_no_index:
        da.assert_eq(expr.apply(self.dataset_xarray_dask, compute=False), expected_dask.data)
    xr.testing.assert_equal(expr.apply(self.dataset_xarray_dask, keep_index=True, compute=False), expected_dask)
    if not skip_no_index:
        np.testing.assert_equal(expr.apply(self.dataset_xarray_dask, compute=True), expected_dask.data.compute())
    xr.testing.assert_equal(expr.apply(self.dataset_xarray_dask, keep_index=True, compute=True), expected_dask.compute())