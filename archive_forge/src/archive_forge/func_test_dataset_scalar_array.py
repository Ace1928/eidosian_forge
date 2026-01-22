import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_scalar_array(self):
    ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
    self.assertEqual(ds.array(), np.column_stack([np.ones(10), np.arange(10)]))