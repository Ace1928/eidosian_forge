import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_iloc_slice_rows_index_cols(self):
    sliced = self.table.iloc[1:2, 2]
    table = Dataset({'Weight': self.weight[1:2]}, kdims=[], vdims=self.vdims[:1])
    self.assertEqual(sliced, table)