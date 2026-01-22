import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_gridded_dtypes(self):
    ds = self.dataset_grid
    self.assertEqual(ds.interface.dtype(ds, 'x'), np.dtype(int))
    self.assertEqual(ds.interface.dtype(ds, 'y'), np.float64)
    self.assertEqual(ds.interface.dtype(ds, 'z'), np.dtype(int))