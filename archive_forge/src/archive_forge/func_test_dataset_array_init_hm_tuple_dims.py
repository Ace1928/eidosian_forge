import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_array_init_hm_tuple_dims(self):
    dataset = Dataset(np.column_stack([self.xs, self.xs_2]), kdims=[('x', 'X')], vdims=[('x2', 'X2')])
    self.assertTrue(isinstance(dataset.data, self.data_type))