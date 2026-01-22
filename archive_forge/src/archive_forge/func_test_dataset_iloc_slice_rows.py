import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_iloc_slice_rows(self):
    sliced = self.table.iloc[1:2]
    table = Dataset({'Gender': self.gender[1:2], 'Age': self.age[1:2], 'Weight': self.weight[1:2], 'Height': self.height[1:2]}, kdims=self.kdims, vdims=self.vdims)
    self.assertEqual(sliced, table)