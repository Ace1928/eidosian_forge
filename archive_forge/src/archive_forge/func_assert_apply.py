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
def assert_apply(self, expr, expected, skip_dask=False, skip_no_index=False):
    if np.isscalar(expected):
        self.assertEqual(expr.apply(self.dataset, keep_index=False), expected)
        self.assertEqual(expr.apply(self.dataset, keep_index=True), expected)
        if dd is None:
            return
        self.assertEqual(expr.apply(self.dataset_dask, keep_index=False), expected)
        self.assertEqual(expr.apply(self.dataset_dask, keep_index=True), expected)
        return
    self.assertIsInstance(expected, pd.Series)
    if not skip_no_index:
        np.testing.assert_equal(expr.apply(self.dataset), expected.values)
    pd.testing.assert_series_equal(expr.apply(self.dataset, keep_index=True), expected, check_names=False)
    if skip_dask or dd is None:
        return
    expected_dask = dd.from_pandas(expected, npartitions=2)
    if not skip_no_index:
        da.assert_eq(expr.apply(self.dataset_dask, compute=False).compute(), expected_dask.values.compute())
    dd.assert_eq(expr.apply(self.dataset_dask, keep_index=True, compute=False), expected_dask, check_names=False)
    if not skip_no_index:
        np.testing.assert_equal(expr.apply(self.dataset_dask, compute=True), expected_dask.values.compute())
    pd.testing.assert_series_equal(expr.apply(self.dataset_dask, keep_index=True, compute=True), expected_dask.compute(), check_names=False)