import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dataframe_init_ht_alias(self):
    """Tests support for heterogeneous DataFrames"""
    dataset = Dataset(pd.DataFrame({'x': self.xs, 'y': self.ys}), kdims=[('x', 'X')], vdims=[('y', 'Y')])
    self.assertTrue(isinstance(dataset.data, self.data_type))