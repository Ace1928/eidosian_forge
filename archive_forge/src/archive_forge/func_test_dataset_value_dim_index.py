import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_value_dim_index(self):
    row = self.table[:, :, 'Weight']
    indexed = Dataset({'Gender': ['M', 'M', 'F'], 'Age': [10, 16, 12], 'Weight': [15, 18, 10]}, kdims=self.kdims, vdims=self.vdims[:1])
    self.assertEqual(row, indexed)